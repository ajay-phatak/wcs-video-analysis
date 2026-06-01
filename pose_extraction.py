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

# Appearance re-ID parameters — keep each dancer's ID locked to the same person
# through slot crossings, where nearest-neighbour position alone is ambiguous.
APP_WEIGHT = 2.0          # weight of colour distance vs body-height-normalised spatial distance
HIST_BINS  = [8, 4, 8]    # hue × saturation × VALUE bins for the torso histogram.
                          # Value (brightness) is essential: the discriminating feature
                          # for low-saturation outfits (black top vs white/grey shirt) is
                          # brightness, which hue–saturation alone cannot separate.


def _hip_center(kps: np.ndarray) -> np.ndarray:
    lh, rh = kps[KP["left_hip"]], kps[KP["right_hip"]]
    return np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])


def _torso_sig(img: np.ndarray, kps: np.ndarray):
    """Hue-saturation histogram of the torso region — a clothing-colour signature
    used to keep dancer identities stable through crossings. Returns None when the
    torso keypoints are too sparse/small to sample reliably."""
    pts = [kps[KP[n]][:2] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
           if kps[KP[n]][2] >= CONF_THRESHOLD]
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    h, w = img.shape[:2]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    px, py = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv  = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HIST_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _app_dist(a, b) -> float:
    """Bhattacharyya distance in [0,1] between two colour signatures; 0 = identical.
    Returns 0 when either signature is missing so appearance simply doesn't vote."""
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


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
            img     = result.orig_img        # original BGR frame for colour sampling
            # Sort detections by confidence descending so we pick the
            # two most confident when more than 2 people are detected
            order = np.argsort(scores)[::-1]
            for i in order[:MAX_DANCERS]:
                kps = kps_all[i]
                dets.append({"kps": kps, "center": _hip_center(kps),
                             "sig": _torso_sig(img, kps)})
        raw_frames.append({"frame_idx": frame_idx,
                            "time_sec":  frame_idx / fps,
                            "dets":      dets})

    # --- Pass 2: identity assignment with FROZEN appearance anchors ------
    # The two dancers are separated up-front into two fixed colour anchors built
    # from all "clean" frames (both dancers near full size), then every frame is
    # assigned against those frozen anchors plus spatial continuity. Freezing the
    # anchors — rather than updating a per-frame EMA reference — is what makes this
    # robust: a single bad frame (a reflection at the walk-on, or one swap during an
    # occlusion) can't poison the reference and lock the identities swapped.
    frames_out = []

    def _scale(dets):
        hs = [body_height(d["kps"]) for d in dets if body_height(d["kps"]) > 10]
        return float(np.mean(hs)) if hs else 100.0

    def _avg_sig(sigs):
        if not sigs:
            return None
        m = np.mean(np.stack(sigs), axis=0).astype(np.float32)
        cv2.normalize(m, m, 0, 1, cv2.NORM_MINMAX)
        return m

    # Clean two-person frames: both detections near full size + both have a signature
    all_bh = [body_height(d["kps"]) for rec in raw_frames for d in rec["dets"]]
    med_bh = float(np.median(all_bh)) if all_bh else 0.0
    clean = [rec["dets"] for rec in raw_frames
             if len(rec["dets"]) == 2 and med_bh > 0
             and body_height(rec["dets"][0]["kps"]) >= 0.6 * med_bh
             and body_height(rec["dets"][1]["kps"]) >= 0.6 * med_bh
             and rec["dets"][0]["sig"] is not None
             and rec["dets"][1]["sig"] is not None]

    # Build two anchors by constrained 2-clustering of the clean-frame pairs: the two
    # detections in any frame are different people, so each pair is split across the two
    # clusters. Seeded by the most-different-looking pair, then refined a few passes.
    anchor = {1: None, 2: None}
    if len(clean) >= 3:
        pairs = [(d[0]["sig"], d[1]["sig"], float(d[0]["center"][0]), float(d[1]["center"][0]))
                 for d in clean]
        seed = max(pairs, key=lambda p: _app_dist(p[0], p[1]))
        a_sig = seed[0] if seed[2] <= seed[3] else seed[1]      # left det of seed → cluster A
        b_sig = seed[1] if seed[2] <= seed[3] else seed[0]
        for _ in range(3):
            a_acc, b_acc = [], []
            for s0, s1, _x0, _x1 in pairs:
                if _app_dist(s0, a_sig) + _app_dist(s1, b_sig) <= \
                   _app_dist(s0, b_sig) + _app_dist(s1, a_sig):
                    a_acc.append(s0); b_acc.append(s1)
                else:
                    a_acc.append(s1); b_acc.append(s0)
            a_sig, b_sig = _avg_sig(a_acc), _avg_sig(b_acc)
        # Map clusters A/B → Dancer 1/2 by which identity is on the LEFT at the
        # first clean frame (keeps the "left = Dancer 1" convention).
        s0, s1, x0, x1 = pairs[0]
        det0_is_a = (_app_dist(s0, a_sig) + _app_dist(s1, b_sig)
                     <= _app_dist(s0, b_sig) + _app_dist(s1, a_sig))
        left_is_a = det0_is_a if x0 <= x1 else (not det0_is_a)
        anchor[1], anchor[2] = (a_sig, b_sig) if left_is_a else (b_sig, a_sig)

    use_app = anchor[1] is not None and anchor[2] is not None

    last_center = {1: None, 2: None}

    def _cost(det, did, scale):
        spatial = (np.linalg.norm(det["center"] - last_center[did]) / scale
                   if last_center[did] is not None else 0.0)
        appear  = APP_WEIGHT * _app_dist(det["sig"], anchor[did]) if use_app else 0.0
        return spatial + appear

    for rec in raw_frames:
        dets = rec["dets"]
        dancers: dict[int, np.ndarray] = {}
        scale = _scale(dets)

        if len(dets) == 0:
            pass  # both missing this frame

        elif len(dets) == 1:
            det = dets[0]
            if last_center[1] is None and last_center[2] is None and not use_app:
                did = 1
            else:
                did = 1 if _cost(det, 1, scale) <= _cost(det, 2, scale) else 2
            dancers[did]     = det["kps"]
            last_center[did] = det["center"]

        else:  # 2 detections
            d0, d1 = dets[0], dets[1]
            if (last_center[1] is None and last_center[2] is None) and not use_app:
                # No anchors and nothing seen yet: label by horizontal position
                lo, hi = (d0, d1) if d0["center"][0] <= d1["center"][0] else (d1, d0)
                dancers[1], dancers[2]         = lo["kps"], hi["kps"]
                last_center[1], last_center[2] = lo["center"], hi["center"]
            else:
                cost_a = _cost(d0, 1, scale) + _cost(d1, 2, scale)   # d0→1, d1→2
                cost_b = _cost(d0, 2, scale) + _cost(d1, 1, scale)   # d0→2, d1→1
                if cost_a <= cost_b:
                    dancers[1], dancers[2]         = d0["kps"], d1["kps"]
                    last_center[1], last_center[2] = d0["center"], d1["center"]
                else:
                    dancers[1], dancers[2]         = d1["kps"], d0["kps"]
                    last_center[1], last_center[2] = d1["center"], d0["center"]

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
