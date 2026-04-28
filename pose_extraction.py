#!/usr/bin/env python3
"""
Pose Extraction
===============
Extracts body keypoints from a dance video for up to two dancers using YOLOv8-pose
with built-in ByteTrack tracking to maintain consistent dancer IDs across frames.

Keypoint indices (COCO 17-point):
  0  nose (head)
  1  left_eye,   2  right_eye
  3  left_ear,   4  right_ear
  5  left_shoulder,  6  right_shoulder
  7  left_elbow,     8  right_elbow
  9  left_wrist,    10  right_wrist
 11  left_hip,      12  right_hip
 13  left_knee,     14  right_knee
 15  left_ankle,    16  right_ankle

Usage:
    python pose_extraction.py video.mp4
    python pose_extraction.py video.mp4 --out poses.npy
    python pose_extraction.py video.mp4 --conf 0.4 --skip 2
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint index map
KP = {
    "head": 0,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

CONF_THRESHOLD = 0.3
MAX_DANCERS = 2


def _hip_center(kps: np.ndarray) -> np.ndarray:
    lh, rh = kps[KP["left_hip"]], kps[KP["right_hip"]]
    return np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])


def extract_poses(video_path: str, conf: float = CONF_THRESHOLD, frame_skip: int = 1,
                  model_name: str = "yolov8n-pose.pt") -> dict:
    """
    Detect all people per frame with YOLOv8-pose, then assign the top-2
    detections to dancer 1 / dancer 2 using greedy nearest-neighbour
    assignment from each frame to the next.

    This completely bypasses ByteTrack IDs, which fragment into hundreds of
    short-lived segments when people occlude each other.  Because we know
    there are exactly 2 dancers, per-frame 2×2 assignment gives near-100%
    coverage whenever both are visible.

    Dancer labelling: in the first 2-person frame, dancer 1 is the person
    whose hip centre has the smaller X coordinate (left in frame).
    """
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- Pass 1: collect per-frame detections (ignore track IDs) --------
    raw_frames: list[dict] = []   # {frame_idx, time_sec, dets: [{kps, center}]}

    for result in model(
        source=video_path,
        stream=True,
        conf=conf,
        verbose=False,
        vid_stride=frame_skip,
    ):
        frame_idx = getattr(result, "frame", len(raw_frames)) * frame_skip
        dets = []
        if result.keypoints is not None and result.boxes is not None:
            kps_all = result.keypoints.data.cpu().numpy()
            scores  = result.boxes.conf.cpu().numpy()
            # Sort detections by confidence descending so we pick the
            # two most confident when more than 2 people are detected
            order = np.argsort(scores)[::-1]
            for i in order[:MAX_DANCERS]:
                kps = kps_all[i]
                dets.append({"kps": kps, "center": _hip_center(kps)})
        raw_frames.append({"frame_idx": frame_idx,
                            "time_sec":  frame_idx / fps,
                            "dets":      dets})

    # --- Pass 2: per-frame 2-person assignment --------------------------
    # State: last known centre for each dancer ID (None = not yet seen)
    last_center = {1: None, 2: None}
    frames_out  = []

    for rec in raw_frames:
        dets = rec["dets"]
        dancers: dict[int, np.ndarray] = {}

        if len(dets) == 0:
            pass  # both missing this frame

        elif len(dets) == 1:
            # Assign to whichever dancer is closer (or dancer 1 if neither seen yet)
            c = dets[0]["center"]
            if last_center[1] is None and last_center[2] is None:
                did = 1
            elif last_center[1] is None:
                did = 2
            elif last_center[2] is None:
                did = 1
            else:
                d1 = np.linalg.norm(c - last_center[1])
                d2 = np.linalg.norm(c - last_center[2])
                did = 1 if d1 <= d2 else 2
            dancers[did] = dets[0]["kps"]
            last_center[did] = dets[0]["center"]

        else:  # 2 detections
            c0, c1 = dets[0]["center"], dets[1]["center"]

            if last_center[1] is None and last_center[2] is None:
                # First 2-person frame: label by horizontal position
                if c0[0] <= c1[0]:
                    dancers[1], dancers[2] = dets[0]["kps"], dets[1]["kps"]
                    last_center[1], last_center[2] = c0, c1
                else:
                    dancers[1], dancers[2] = dets[1]["kps"], dets[0]["kps"]
                    last_center[1], last_center[2] = c1, c0
            else:
                # Use whichever centres we have; fall back to spatial order if one missing
                ref1 = last_center[1] if last_center[1] is not None else last_center[2]
                ref2 = last_center[2] if last_center[2] is not None else last_center[1]

                # 2×2 cost matrix, solve greedily (optimal for 2×2)
                d00 = np.linalg.norm(c0 - ref1)
                d01 = np.linalg.norm(c0 - ref2)
                d10 = np.linalg.norm(c1 - ref1)
                d11 = np.linalg.norm(c1 - ref2)

                if d00 + d11 <= d01 + d10:
                    dancers[1], dancers[2] = dets[0]["kps"], dets[1]["kps"]
                    last_center[1], last_center[2] = c0, c1
                else:
                    dancers[1], dancers[2] = dets[1]["kps"], dets[0]["kps"]
                    last_center[1], last_center[2] = c1, c0

        frames_out.append({
            "frame_idx": rec["frame_idx"],
            "time_sec":  rec["time_sec"],
            "dancers":   dancers,
        })

    d1_count = sum(1 for f in frames_out if 1 in f["dancers"])
    d2_count = sum(1 for f in frames_out if 2 in f["dancers"])
    both     = sum(1 for f in frames_out if 1 in f["dancers"] and 2 in f["dancers"])
    print(f"  Dancer 1: {d1_count} frames  |  Dancer 2: {d2_count} frames  |  Both: {both} frames")

    return {
        "fps":          fps,
        "frame_count":  total,
        "width":        width,
        "height":       height,
        "dancer_ids":   [1, 2],
        "frames":       frames_out,
    }


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def get_kp(kps: np.ndarray, name: str) -> np.ndarray:
    """Return (x, y, conf) for a named keypoint."""
    return kps[KP[name]]


def get_center(kps: np.ndarray) -> np.ndarray:
    """Hip midpoint — used as the dancer's centre of mass proxy."""
    lh = kps[KP["left_hip"]]
    rh = kps[KP["right_hip"]]
    return np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, min(lh[2], rh[2])])


def body_height(kps: np.ndarray) -> float:
    """Approximate body height in pixels for normalisation (head → mid-ankle)."""
    head   = kps[KP["head"]]
    la, ra = kps[KP["left_ankle"]], kps[KP["right_ankle"]]
    ankle_y = (la[1] + ra[1]) / 2
    return max(abs(ankle_y - head[1]), 1.0)


def torso_angle(kps: np.ndarray) -> float:
    """
    Angle (degrees) of the torso lean from vertical.
    Positive = leaning right, negative = leaning left.
    Computed as the angle of the head-to-hip-midpoint vector.
    """
    head   = kps[KP["head"]]
    center = get_center(kps)
    dx = center[0] - head[0]
    dy = center[1] - head[1]   # positive = downward in image coords
    return float(np.degrees(np.arctan2(dx, dy)))


def shoulder_angle(kps: np.ndarray) -> float:
    """Angle (degrees) of the shoulder line from horizontal."""
    ls = kps[KP["left_shoulder"]]
    rs = kps[KP["right_shoulder"]]
    return float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))


def hip_angle(kps: np.ndarray) -> float:
    """Angle (degrees) of the hip line from horizontal."""
    lh = kps[KP["left_hip"]]
    rh = kps[KP["right_hip"]]
    return float(np.degrees(np.arctan2(rh[1] - lh[1], rh[0] - lh[0])))


# ---------------------------------------------------------------------------
# Serialisation helpers (numpy arrays → JSON-safe)
# ---------------------------------------------------------------------------

def poses_to_serialisable(data: dict) -> dict:
    """Convert numpy arrays and numpy scalars to JSON-safe Python types."""
    def _to_py(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = {k: ([int(i) for i in v] if k == "dancer_ids" else _to_py(v))
           for k, v in data.items() if k != "frames"}
    out["frames"] = []
    for f in data["frames"]:
        out["frames"].append({
            "frame_idx": _to_py(f["frame_idx"]),
            "time_sec":  _to_py(f["time_sec"]),
            "dancers":   {str(int(k)): v.tolist() for k, v in f["dancers"].items()},
        })
    return out


def poses_from_serialisable(data: dict) -> dict:
    """Restore numpy arrays from JSON-loaded dict."""
    frames = []
    for f in data["frames"]:
        frames.append({
            "frame_idx": f["frame_idx"],
            "time_sec":  f["time_sec"],
            "dancers":   {int(k): np.array(v) for k, v in f["dancers"].items()},
        })
    return {**data, "frames": frames}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from a dance video.")
    parser.add_argument("video",       help="Path to video file")
    parser.add_argument("--out",       default=None, help="Output path (.json). Defaults to <video>.poses.json")
    parser.add_argument("--conf",      type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--skip",      type=int,   default=1, help="Process every Nth frame")
    parser.add_argument("--model",     default="yolov8n-pose.pt",
                        help="YOLOv8 pose model (n=fastest, s/m/l/x=more accurate)")
    args = parser.parse_args()

    video_path = args.video
    out_path   = args.out or str(Path(video_path).with_suffix(".poses.json"))

    print(f"Extracting poses from: {video_path}")
    data = extract_poses(video_path, conf=args.conf, frame_skip=args.skip, model_name=args.model)

    n_frames  = len(data["frames"])
    n_dancers = len(data["dancer_ids"])
    print(f"  {n_frames} frames processed  |  {n_dancers} dancer(s) tracked  |  IDs: {data['dancer_ids']}")

    with open(out_path, "w") as fh:
        json.dump(poses_to_serialisable(data), fh)
    print(f"  Poses saved to: {out_path}")


if __name__ == "__main__":
    main()
