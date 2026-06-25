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

# Crowd handling: keep up to MAX_KEEP detections per frame before identity matching
# (was hard-capped at the top-2 by confidence, which discarded the target couple in a
# crowded room before appearance matching could even run). With a seed, every frame's
# detections are matched against the two seeded anchors instead of assuming top-2 = couple.
MAX_KEEP = 12
# Max Bhattacharyya colour distance to accept a detection as a seeded dancer (lower =
# stricter). Beyond this the detection is "not this person" → dancer left missing.
APP_GATE = 0.55
# When a detection has no colour signature, accept it as a seeded dancer only if within
# this many body-heights of that dancer's last known position (motion continuity).
SPATIAL_GATE = 1.75

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
                  model_name: str = "yolov8m-pose.pt",
                  seed_frame_idx: int | None = None,
                  seed_points: list | None = None) -> dict:
    """
    Detect people per frame with YOLOv8-pose and assign two of them to dancer 1 /
    dancer 2 using a frozen-appearance + spatial-continuity cost.

    Two identity modes:
      • Unseeded (default, clean 2-person footage): keep the top-2 detections per
        frame and split them into two colour anchors built from "clean" frames. Good
        when the only two people on screen are the couple.
      • Seeded (`seed_points`, for crowded footage): keep ALL detections (up to
        MAX_KEEP) and match every frame against two anchors built from the two people
        the caller pointed at in `seed_frame_idx`. Detections that match neither
        anchor (other couples) are ignored; when the target is occluded the dancer is
        left MISSING for that frame rather than grabbing a stranger. `seed_points` is
        [(x1, y1), (x2, y2)] in original-frame pixels → dancer 1, dancer 2 respectively.

    ByteTrack IDs are bypassed either way (they fragment under occlusion).
    Dancer labelling (unseeded): in the first 2-person frame, dancer 1 is the person
    whose hip centre has the smaller X coordinate (left in frame). Seeded: dancer 1 is
    the first seed point.
    """
    model = YOLO(model_name)
    seeded = bool(seed_points) and len(seed_points) == 2

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
            # Sort detections by confidence descending. Keep up to MAX_KEEP so a
            # seeded couple in a crowd survives into identity matching (unseeded mode
            # still uses only the top-2 in Pass 2).
            order = np.argsort(scores)[::-1]
            for i in order[:MAX_KEEP]:
                kps = kps_all[i]
                dets.append({"kps": kps, "center": _hip_center(kps),
                             "sig": _torso_sig(img, kps), "score": float(scores[i])})
        raw_frames.append({"frame_idx": frame_idx,
                            "time_sec":  frame_idx / fps,
                            "dets":      dets})

    # --- Pass 2: identity assignment with FROZEN appearance anchors ------
    # Each dancer is pinned to a fixed colour anchor, then every frame is assigned
    # against those frozen anchors plus spatial continuity. Freezing the anchors —
    # rather than updating a per-frame EMA reference — is what makes this robust: a
    # single bad frame (a reflection at the walk-on, or one swap during an occlusion)
    # can't poison the reference and lock the identities swapped.
    #   Unseeded: anchors from constrained 2-clustering of clean 2-person frames.
    #   Seeded:   anchors from the two people pointed at in the seed frame, refined
    #             once from their matched detections; all detections are matched and
    #             non-matching ones (other couples) are ignored.
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

    anchor = {1: None, 2: None}
    seed_center = {1: None, 2: None}

    if seeded:
        # Anchor each dancer to the detection nearest its seed point in the seed frame.
        target = min(raw_frames, key=lambda r: abs(r["frame_idx"] - (seed_frame_idx or 0)))
        for did, pt in zip((1, 2), seed_points):
            if target["dets"]:
                best = min(target["dets"],
                           key=lambda d: float(np.hypot(d["center"][0] - pt[0],
                                                         d["center"][1] - pt[1])))
                anchor[did]      = best["sig"]
                seed_center[did] = np.asarray(best["center"], dtype=float)
    else:
        # Clean two-person frames: both detections near full size + both have a signature
        all_bh = [body_height(d["kps"]) for rec in raw_frames for d in rec["dets"]]
        med_bh = float(np.median(all_bh)) if all_bh else 0.0
        clean = [rec["dets"][:MAX_DANCERS] for rec in raw_frames
                 if len(rec["dets"]) == 2 and med_bh > 0
                 and body_height(rec["dets"][0]["kps"]) >= 0.6 * med_bh
                 and body_height(rec["dets"][1]["kps"]) >= 0.6 * med_bh
                 and rec["dets"][0]["sig"] is not None
                 and rec["dets"][1]["sig"] is not None]
        # Constrained 2-clustering of clean-frame pairs: the two detections in any frame
        # are different people, so each pair is split across the two clusters. Seeded by
        # the most-different-looking pair, then refined a few passes.
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

    def _cost(det, did, scale, lc):
        spatial = (np.linalg.norm(det["center"] - lc[did]) / max(scale, 1.0)
                   if lc[did] is not None else 0.0)
        appear  = APP_WEIGHT * _app_dist(det["sig"], anchor[did]) if anchor[did] is not None else 0.0
        return spatial + appear

    def _passes_gate(det, did, scale, lc):
        # Reject detections that are clearly not the seeded dancer.
        if anchor[did] is not None and det["sig"] is not None:
            return _app_dist(det["sig"], anchor[did]) <= APP_GATE
        if lc[did] is not None:        # no colour info → require motion continuity
            return np.linalg.norm(det["center"] - lc[did]) / max(scale, 1.0) <= SPATIAL_GATE
        return False                   # first sighting, no colour, no position → unsafe

    def _assign_seeded(dets, scale, lc):
        """Match ALL detections to the two anchors. Returns {did: det_index}; a dancer
        may be left unassigned (occluded), and no detection is shared by both."""
        cands = sorted(((_cost(det, did, scale, lc), did, j)
                        for j, det in enumerate(dets) for did in (1, 2)),
                       key=lambda c: c[0])
        out, used = {}, set()
        for _c, did, j in cands:
            if did in out or j in used:
                continue
            if not _passes_gate(dets[j], did, scale, lc):
                continue
            out[did] = j; used.add(j)
        return out

    if seeded:
        # One refinement pass: rough-assign across the clip to gather each dancer's
        # matched colour signatures, average them into sturdier anchors, then re-run.
        lc = {1: seed_center[1], 2: seed_center[2]}
        sig_acc = {1: [], 2: []}
        for rec in raw_frames:
            scale = _scale(rec["dets"])
            for did, j in _assign_seeded(rec["dets"], scale, lc).items():
                lc[did] = np.asarray(rec["dets"][j]["center"], dtype=float)
                if rec["dets"][j]["sig"] is not None:
                    sig_acc[did].append(rec["dets"][j]["sig"])
        for did in (1, 2):
            avg = _avg_sig(sig_acc[did])
            if avg is not None:
                anchor[did] = avg

    last_center = {1: seed_center[1], 2: seed_center[2]} if seeded else {1: None, 2: None}

    for rec in raw_frames:
        dets = rec["dets"]
        dancers: dict[int, np.ndarray] = {}
        scale = _scale(dets)

        if seeded:
            for did, j in _assign_seeded(dets, scale, last_center).items():
                dancers[did]     = dets[j]["kps"]
                last_center[did] = np.asarray(dets[j]["center"], dtype=float)

        else:
            dets = dets[:MAX_DANCERS]   # unseeded: only the top-2 by confidence
            if len(dets) == 0:
                pass  # both missing this frame

            elif len(dets) == 1:
                det = dets[0]
                if last_center[1] is None and last_center[2] is None and not use_app:
                    did = 1
                else:
                    did = 1 if _cost(det, 1, scale, last_center) <= _cost(det, 2, scale, last_center) else 2
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
                    cost_a = _cost(d0, 1, scale, last_center) + _cost(d1, 2, scale, last_center)
                    cost_b = _cost(d0, 2, scale, last_center) + _cost(d1, 1, scale, last_center)
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
    mode = "seeded" if seeded else "top-2"
    print(f"  [{mode}, model={model_name}]  Dancer 1: {d1_count} frames  |  "
          f"Dancer 2: {d2_count} frames  |  Both: {both} frames")
    if seeded and both < 0.5 * len(frames_out):
        print("  NOTE: target couple matched in <50% of frames — they may be heavily "
              "occluded, or the seed points/anchors are off. Try a clearer seed frame.")

    return {
        "fps":          fps,
        "frame_count":  total,
        "width":        width,
        "height":       height,
        "dancer_ids":   [1, 2],
        "model":        model_name,
        "seeded":       seeded,
        "frames":       frames_out,
    }


def detect_single_frame(video_path: str, t_sec: float,
                        model_name: str = "yolov8m-pose.pt",
                        conf: float = CONF_THRESHOLD):
    """Detect all people in ONE frame (at t_sec seconds) for the crowd-mode seed step.

    Returns (frame_idx, bgr_image, dets) where dets is a confidence-sorted list of
    {center: [x,y], box: [x0,y0,x1,y1], conf: float, kps: ndarray}. The caller labels
    the people so the user can point out which two are the target couple.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {t_sec}s ({video_path})")

    res = YOLO(model_name)(img, conf=conf, verbose=False)[0]
    dets = []
    if res.keypoints is not None and res.boxes is not None:
        kps_all = res.keypoints.data.cpu().numpy()
        boxes   = res.boxes.xyxy.cpu().numpy()
        scores  = res.boxes.conf.cpu().numpy()
        for i in np.argsort(scores)[::-1]:
            dets.append({"center": _hip_center(kps_all[i]),
                         "box":    boxes[i],
                         "conf":   float(scores[i]),
                         "kps":    kps_all[i]})
    return frame_idx, img, dets


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
    parser.add_argument("--model",     default="yolov8m-pose.pt",
                        help="YOLOv8 pose model (n=fastest, s/m/l/x=more accurate). "
                             "Default m; use l/x for max accuracy on small/crowded figures.")
    parser.add_argument("--seed-frame", type=int, default=None,
                        help="Frame index of the seed frame for crowd mode (pair with --seed-me/--seed-partner).")
    parser.add_argument("--seed-me",      default=None, help="'x,y' pixel point on dancer 1 in the seed frame.")
    parser.add_argument("--seed-partner", default=None, help="'x,y' pixel point on dancer 2 in the seed frame.")
    args = parser.parse_args()

    video_path = args.video
    out_path   = args.out or str(Path(video_path).with_suffix(".poses.json"))

    seed_points = None
    if args.seed_me and args.seed_partner:
        seed_points = [tuple(float(v) for v in args.seed_me.split(",")),
                       tuple(float(v) for v in args.seed_partner.split(","))]

    print(f"Extracting poses from: {video_path}")
    data = extract_poses(video_path, conf=args.conf, frame_skip=args.skip, model_name=args.model,
                         seed_frame_idx=args.seed_frame, seed_points=seed_points)

    n_frames  = len(data["frames"])
    n_dancers = len(data["dancer_ids"])
    print(f"  {n_frames} frames processed  |  {n_dancers} dancer(s) tracked  |  IDs: {data['dancer_ids']}")

    with open(out_path, "w") as fh:
        json.dump(poses_to_serialisable(data), fh)
    print(f"  Poses saved to: {out_path}")


if __name__ == "__main__":
    main()
