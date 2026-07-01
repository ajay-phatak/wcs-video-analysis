#!/usr/bin/env python3
"""
3D Pose Lifting (Pass 3)
========================
Lifts each dancer's 2D keypoint sequence to 3D with VideoPose3D (Facebook
Research, CC-BY-NC-4.0 — model definition vendored in videopose3d_model.py,
pretrained on Human3.6M with Detectron COCO-17 2D inputs).

Why: the 2D joint angles the metrics use are view-dependent — your camera
angle rarely matches a pro clip's, so 2D angle comparisons carry a hidden
camera bias. The lifted 3D joints are root-relative in camera space; joint
ANGLES computed from them are rotation-invariant, i.e. comparable across
clips filmed from different positions.

What it adds to the poses JSON (2D data is untouched):
  frames[i]["dancers3d"][did]  →  17×3 metres, root(pelvis)-relative
  top-level "kps3d_format": "h36m17", "lift_model", "kps3d_space"

H36M-17 joint order (differs from COCO!):
   0 pelvis/root   1 r_hip   2 r_knee   3 r_ankle   4 l_hip   5 l_knee
   6 l_ankle       7 spine   8 thorax   9 neck     10 head   11 l_shoulder
  12 l_elbow      13 l_wrist 14 r_shoulder 15 r_elbow 16 r_wrist

Notes / limitations:
  * Root-relative: global travel and absolute rise/fall are NOT in the 3D
    output (the pelvis is the origin every frame) — keep using 2D for those.
  * Trained at 50 fps on studio mocap; 25-60 fps in-the-wild footage works
    but treat absolute depths as approximate. Angles are the reliable read.
  * Input quality matters: run pose_refine.py first (lifting pass-1 keypoints
    propagates their jitter into 3D).

Usage:
    python pose_lift.py path/to/clip_poses.json            # augment in place
    python pose_lift.py path/to/clip.mp4                   # uses <stem>_poses.json
    python pose_lift.py path/to/clip.mp4 --out other.json
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

import pose_refine as pr

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
CHECKPOINT_DIR = Path.home() / ".cache" / "videopose3d"

# Model architecture of the released checkpoint (VideoPose3D run.py defaults):
# 243-frame receptive field, 5 blocks of width-3 dilated convolutions.
FILTER_WIDTHS = [3, 3, 3, 3, 3]
CHANNELS = 1024

# COCO-17 left/right columns (input flip augmentation)
KPS_LEFT  = [1, 3, 5, 7, 9, 11, 13, 15]
KPS_RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]
# H36M-17 left/right joints (output un-flip)
JOINTS_LEFT  = [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]

H36M_KP = {
    "pelvis": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
    "left_hip": 4, "left_knee": 5, "left_ankle": 6, "spine": 7,
    "thorax": 8, "neck": 9, "head": 10, "left_shoulder": 11,
    "left_elbow": 12, "left_wrist": 13, "right_shoulder": 14,
    "right_elbow": 15, "right_wrist": 16,
}

# 2D keypoints below this confidence are treated as missing and interpolated
# from neighbouring frames before lifting (a (0,0) joint would poison the lift).
MIN_KP_CONF = 0.1


def _get_checkpoint() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / Path(CHECKPOINT_URL).name
    if not path.exists():
        print(f"  Downloading VideoPose3D checkpoint → {path}")
        urllib.request.urlretrieve(CHECKPOINT_URL, path)
    return path


def _load_model():
    import torch
    from videopose3d_model import TemporalModel
    model = TemporalModel(17, 2, 17, filter_widths=FILTER_WIDTHS,
                          causal=False, dropout=0.25, channels=CHANNELS)
    # weights_only=True: tensors-only deserialisation, no arbitrary pickle code
    ckpt = torch.load(_get_checkpoint(), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_pos"])
    model.eval()
    return model


def _interp_gaps(seq: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-joint linear interpolation over time where valid is False.
    seq: (T, J, 2), valid: (T, J). Edges are held at the nearest valid value."""
    T = seq.shape[0]
    t = np.arange(T)
    out = seq.copy()
    for j in range(seq.shape[1]):
        v = valid[:, j]
        if v.all():
            continue
        if not v.any():          # joint never seen — leave zeros
            continue
        for c in range(2):
            out[:, j, c] = np.interp(t, t[v], seq[v, j, c])
    return out


def lift_poses(pose_data: dict) -> dict:
    """Add root-relative 3D joints for every tracked dancer. `pose_data` is the
    numpy-form dict from pose_refine.load_pass1 (2D kps may be 17 or 26 rows;
    only the first 17 COCO rows are used)."""
    import torch

    model = _load_model()
    pad = (model.receptive_field() - 1) // 2

    frames = pose_data["frames"]
    w, h = float(pose_data["width"]), float(pose_data["height"])
    dids = sorted({d for f in frames for d in f["dancers"]})

    lifted: dict[int, dict[int, np.ndarray]] = {}   # did -> {frame_pos: (17,3)}
    t0 = time.time()

    for did in dids:
        pos = [i for i, f in enumerate(frames) if did in f["dancers"]]
        if len(pos) < 2:
            continue
        span = range(pos[0], pos[-1] + 1)

        seq = np.zeros((len(span), 17, 2), dtype=np.float32)
        valid = np.zeros((len(span), 17), dtype=bool)
        for i in span:
            k = i - pos[0]
            d = frames[i]["dancers"].get(did)
            if d is not None:
                kps = np.asarray(d, dtype=np.float32)[:17]
                seq[k] = kps[:, :2]
                valid[k] = kps[:, 2] >= MIN_KP_CONF
        seq = _interp_gaps(seq, valid)

        # VideoPose3D screen-coordinate normalisation
        seq = seq / w * 2.0 - np.array([1.0, h / w], dtype=np.float32)

        inp = np.pad(seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        flip = inp.copy()
        flip[:, :, 0] *= -1
        flip[:, KPS_LEFT + KPS_RIGHT] = flip[:, KPS_RIGHT + KPS_LEFT]

        with torch.no_grad():
            batch = torch.from_numpy(np.stack([inp, flip]))
            out = model(batch).numpy()          # (2, T, 17, 3)
        out[1, :, :, 0] *= -1
        out[1, :, JOINTS_LEFT + JOINTS_RIGHT] = out[1, :, JOINTS_RIGHT + JOINTS_LEFT]
        pose3d = out.mean(axis=0)               # flip-averaged, (T, 17, 3)

        # Attach only to frames where the dancer was actually tracked
        lifted[did] = {i: pose3d[i - pos[0]] for i in pos}
        print(f"  dancer {did}: lifted {len(pos)} frames "
              f"(span {pos[0]}–{pos[-1]}, {time.time()-t0:.0f}s)")

    for i, f in enumerate(frames):
        d3 = {did: lifted[did][i] for did in lifted if i in lifted[did]}
        if d3:
            f["dancers3d"] = d3

    pose_data["kps3d_format"] = "h36m17"
    pose_data["kps3d_space"] = "camera-frame, root-relative, metres"
    pose_data["lift_model"] = "videopose3d-h36m-detectron-coco"
    return pose_data


def save_poses(pose_data: dict, out_path: Path):
    """JSON-safe serialisation including the dancers3d frame key (which
    pose_extraction.poses_to_serialisable doesn't know about)."""
    out = {k: v for k, v in pose_data.items() if k != "frames"}
    out["frames"] = []
    for f in pose_data["frames"]:
        rec = {"frame_idx": int(f["frame_idx"]),
               "time_sec":  float(f["time_sec"]),
               "dancers":   {str(k): np.asarray(v).tolist()
                             for k, v in f["dancers"].items()}}
        if "dancers3d" in f:
            rec["dancers3d"] = {str(k): np.round(np.asarray(v), 4).tolist()
                                for k, v in f["dancers3d"].items()}
        out["frames"].append(rec)
    out_path.write_text(json.dumps(out), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Lift 2D dance poses to 3D (VideoPose3D).")
    ap.add_argument("input", help="Poses JSON, or a video path (uses <stem>_poses.json)")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default: augment the input file in place)")
    args = ap.parse_args()

    inp = Path(args.input)
    poses_path = inp if inp.suffix == ".json" else inp.with_name(inp.stem + "_poses.json")
    if not poses_path.exists():
        raise SystemExit(f"Poses not found: {poses_path}")

    print(f"Lifting to 3D: {poses_path.name}")
    data = pr.load_pass1(poses_path)
    if data.get("keypoint_format") != "halpe26":
        print("  NOTE: input looks like unrefined pass-1 poses — 3D quality is much "
              "better after pose_refine.py.")

    data = lift_poses(data)

    out_path = Path(args.out) if args.out else poses_path
    save_poses(data, out_path)
    print(f"  3D-augmented poses saved to: {out_path}")


if __name__ == "__main__":
    main()
