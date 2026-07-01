#!/usr/bin/env python3
"""
Pose Refinement (Pass 2)
========================
Re-estimates each tracked dancer's keypoints with a top-down RTMPose model in
Halpe-26 format, using the pass-1 output of pose_extraction.py for identity
and rough location.

Why a second pass:
  * Pass 1 (YOLOv8-pose) sees the whole frame at ~640px input. A dancer who is
    150px tall in a spotlight/crowd clip is a thumbnail to the pose head, so
    fine joint positions (ankles, knees) are approximate.
  * Top-down RTMPose is given the dancer's bounding box and crops from the
    ORIGINAL-resolution frame, so the dancer fills the model input regardless
    of how small they are in frame.
  * Halpe-26 adds heels + big/small toes (COCO-17 has no foot keypoints), so
    ankle-lift proxies can become real heel/toe articulation measurements.

Output schema matches pose_extraction.py exactly, except each keypoint array
is (26, 3) instead of (17, 3). Halpe-26 indices 0-16 are identical to COCO-17,
so all existing metrics work unchanged; feet live at 20-25:

  17 head_top   18 neck          19 hip_center
  20 left_big_toe   21 right_big_toe
  22 left_small_toe 23 right_small_toe
  24 left_heel      25 right_heel

Usage:
    python pose_refine.py video.mp4                        # refines <stem>_poses.json
    python pose_refine.py video.mp4 --mode performance     # biggest/most accurate model
    python pose_refine.py video.mp4 --max-frames 300       # quick test on a subset
    python pose_refine.py video.mp4 --preview 40           # overlay png at t=40s
    python pose_refine.py video.mp4 --replace-cache        # write back to <stem>_poses.json
                                                           # (pass-1 kept as *_poses_pass1.json)
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

import pose_extraction as pe

# Halpe-26 extends the COCO-17 map used in pass 1 (indices 0-16 unchanged).
HALPE_KP = {
    **pe.KP,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "left_big_toe": 20,   "right_big_toe": 21,
    "left_small_toe": 22, "right_small_toe": 23,
    "left_heel": 24,      "right_heel": 25,
}

N_HALPE = 26

# RTMPose Halpe-26 checkpoints (ONNX, downloaded and cached by rtmlib on first use).
POSE_MODELS = {
    "lightweight": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",  # noqa
        "input_size": (192, 256),
    },
    "balanced": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",  # noqa
        "input_size": (192, 256),
    },
    "performance": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",  # noqa
        "input_size": (288, 384),
    },
}

# Pass-1 keypoints below this confidence don't contribute to the dancer's box.
BOX_KP_CONF = 0.15
# Box padding as a fraction of box size. RTMPose pads another 1.25x internally,
# so this only needs to cover parts pass 1 missed entirely — mainly the feet,
# which sit below the ankles (COCO-17's lowest points), hence the larger bottom pad.
PAD_X, PAD_TOP, PAD_BOTTOM = 0.08, 0.08, 0.15


def _parse_kps(raw) -> np.ndarray:
    """Keypoints from a cached JSON may be a list OR a numpy string-repr
    (analyze.py saves with json.dumps(default=str)). Same tolerant parsing
    as analyze.py's _normalise_poses."""
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        return raw
    if isinstance(raw, str) or (isinstance(raw, np.ndarray) and raw.ndim == 0):
        nums = [float(x) for x in
                re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))]
        return np.array(nums, dtype=float).reshape(-1, 3)
    arr = np.array(raw, dtype=float)
    return arr.reshape(-1, 3) if arr.ndim == 1 else arr


def load_pass1(poses_path: Path) -> dict:
    """Load a pass-1 poses JSON into numpy form, tolerating both serialisation
    styles (pose_extraction.py lists and analyze.py default=str strings)."""
    data = json.loads(poses_path.read_text(encoding="utf-8"))
    frames = []
    for f in data["frames"]:
        frames.append({
            "frame_idx": int(f["frame_idx"]),
            "time_sec":  float(f["time_sec"]),
            "dancers":   {int(k): _parse_kps(v)
                          for k, v in f.get("dancers", {}).items()},
        })
    return {**data, "frames": frames}


def _load_pose_model(mode: str, device: str):
    from rtmlib import RTMPose
    spec = POSE_MODELS[mode]
    return RTMPose(spec["url"], model_input_size=spec["input_size"],
                   backend="onnxruntime", device=device)


def _bbox_from_kps(kps: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Padded xyxy box around a dancer's confident pass-1 keypoints."""
    pts = kps[kps[:, 2] >= BOX_KP_CONF][:, :2]
    if len(pts) < 4:
        return None
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 8:
        return None
    return np.array([max(0.0, x0 - PAD_X * bw),
                     max(0.0, y0 - PAD_TOP * bh),
                     min(float(w), x1 + PAD_X * bw),
                     min(float(h), y1 + PAD_BOTTOM * bh)])


def _pad_to_halpe(kps: np.ndarray) -> np.ndarray:
    """Extend a (17,3) pass-1 array to (26,3) with zero-confidence extras,
    for frames where refinement was skipped (keeps the output shape uniform)."""
    if kps.shape[0] >= N_HALPE:
        return kps
    out = np.zeros((N_HALPE, 3), dtype=kps.dtype)
    out[:kps.shape[0]] = kps
    return out


def refine_poses(video_path: str, pose_data: dict, mode: str = "balanced",
                 device: str = "cpu", max_frames: int | None = None) -> dict:
    """Run the top-down refinement pass. `pose_data` is the pass-1 dict (numpy
    form, as returned by pe.extract_poses / pe.poses_from_serialisable).
    Returns a new dict with the same schema and (26,3) keypoint arrays."""
    pose_model = _load_pose_model(mode, device)

    frames = pose_data["frames"]
    by_idx = {int(f["frame_idx"]): f for f in frames}
    todo = sorted(by_idx)
    if max_frames is not None:
        todo = todo[:max_frames]
    todo_set = set(todo)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    refined_frames = []
    n_done = n_skipped = 0
    t0 = time.time()
    frame_idx = -1
    last_todo = todo[-1] if todo else -1

    while frame_idx < last_todo:
        ok, img = cap.read()
        frame_idx += 1
        if not ok:
            break
        if frame_idx not in todo_set:
            continue

        rec = by_idx[frame_idx]
        dids = sorted(rec["dancers"])
        boxes, box_dids = [], []
        for did in dids:
            box = _bbox_from_kps(np.asarray(rec["dancers"][did]), w, h)
            if box is not None:
                boxes.append(box)
                box_dids.append(did)

        new_dancers = {}
        if boxes:
            kps26, scores = pose_model(img, bboxes=boxes)
            for i, did in enumerate(box_dids):
                new_dancers[did] = np.concatenate(
                    [kps26[i], scores[i][:, None]], axis=1)
            n_done += len(box_dids)
        # dancers whose box couldn't be formed keep their pass-1 keypoints
        for did in dids:
            if did not in new_dancers:
                new_dancers[did] = _pad_to_halpe(np.asarray(rec["dancers"][did]))
                n_skipped += 1

        refined_frames.append({"frame_idx": rec["frame_idx"],
                               "time_sec":  rec["time_sec"],
                               "dancers":   new_dancers})

        if len(refined_frames) % 200 == 0:
            rate = len(refined_frames) / (time.time() - t0)
            eta = (len(todo) - len(refined_frames)) / max(rate, 1e-6)
            print(f"  refined {len(refined_frames)}/{len(todo)} frames "
                  f"({rate:.1f} f/s, ~{eta/60:.1f} min left)")

    cap.release()

    # When testing on a subset, pass through the un-refined tail unchanged
    # (padded to 26 rows) so the output is still a complete, loadable file.
    done_idx = {f["frame_idx"] for f in refined_frames}
    for f in frames:
        if f["frame_idx"] not in done_idx:
            refined_frames.append({
                "frame_idx": f["frame_idx"],
                "time_sec":  f["time_sec"],
                "dancers":   {d: _pad_to_halpe(np.asarray(k))
                              for d, k in f["dancers"].items()},
            })
    refined_frames.sort(key=lambda f: f["frame_idx"])

    dt = time.time() - t0
    print(f"  [refine, mode={mode}]  {n_done} dancer-poses refined, "
          f"{n_skipped} kept from pass 1  ({dt:.0f}s)")

    out = {k: v for k, v in pose_data.items() if k != "frames"}
    out["frames"] = refined_frames
    out["keypoint_format"] = "halpe26"
    out["refine_model"] = f"rtmpose-halpe26-{mode}"
    out["refined_frame_count"] = len(done_idx)
    return out


# ---------------------------------------------------------------------------
# Preview rendering — pass-1 vs refined skeleton overlay for eyeballing
# ---------------------------------------------------------------------------

_LIMBS = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
_FOOT_LIMBS = [
    ("left_ankle", "left_heel"), ("left_heel", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_heel", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
]


def _draw_skel(img, kps, colour, kp_map, limbs, conf=0.3):
    for a, b in limbs:
        pa, pb = kps[kp_map[a]], kps[kp_map[b]]
        if pa[2] >= conf and pb[2] >= conf:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     colour, 2, cv2.LINE_AA)
    for name in {n for pair in limbs for n in pair}:
        p = kps[kp_map[name]]
        if p[2] >= conf:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, colour, -1, cv2.LINE_AA)


def save_preview(video_path: str, pass1: dict, refined: dict, t_sec: float,
                 out_path: str):
    """Side-by-side frame: pass-1 skeletons (left, red) vs refined (right,
    green, feet in yellow)."""
    fps = refined.get("fps", 30.0)
    target = int(round(t_sec * fps))
    recs1 = {f["frame_idx"]: f for f in pass1["frames"]}
    recs2 = {f["frame_idx"]: f for f in refined["frames"]}
    common = sorted(set(recs1) & set(recs2))
    idx = min(common, key=lambda i: abs(i - target))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}")

    left, right = img.copy(), img.copy()
    for kps in recs1[idx]["dancers"].values():
        _draw_skel(left, np.asarray(kps), (0, 0, 255), pe.KP, _LIMBS)
    for kps in recs2[idx]["dancers"].values():
        kps = np.asarray(kps)
        _draw_skel(right, kps, (0, 255, 0), pe.KP, _LIMBS)
        if kps.shape[0] >= N_HALPE:
            _draw_skel(right, kps, (0, 255, 255), HALPE_KP, _FOOT_LIMBS)

    for im, label in ((left, "pass 1 (YOLOv8)"), (right, "refined (RTMPose halpe26)")):
        cv2.putText(im, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
    combo = np.concatenate([left, right], axis=1)
    cv2.imwrite(out_path, combo)
    print(f"  Preview (frame {idx}, t={idx/fps:.1f}s) saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Top-down Halpe-26 refinement of pass-1 poses.")
    ap.add_argument("video", help="Path to the video file")
    ap.add_argument("--poses", default=None,
                    help="Pass-1 poses JSON (default: <stem>_poses.json next to the video)")
    ap.add_argument("--out", default=None,
                    help="Output JSON (default: <stem>_poses_refined.json)")
    ap.add_argument("--mode", choices=list(POSE_MODELS), default="balanced",
                    help="RTMPose size: lightweight (s) / balanced (m) / performance (x, 384x288)")
    ap.add_argument("--device", default="cpu", help="onnxruntime device: cpu or cuda")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Refine only the first N tracked frames (quick test)")
    ap.add_argument("--preview", type=float, default=None, metavar="T_SEC",
                    help="Also save a pass1-vs-refined overlay png near this timestamp")
    ap.add_argument("--replace-cache", action="store_true",
                    help="Write result to <stem>_poses.json (what analyze.py loads); "
                         "the pass-1 file is kept as <stem>_poses_pass1.json")
    args = ap.parse_args()

    video = Path(args.video)
    poses_path = Path(args.poses) if args.poses else video.with_name(video.stem + "_poses.json")
    if not poses_path.exists():
        raise SystemExit(f"Pass-1 poses not found: {poses_path}\n"
                         f"Run pose_extraction.py (or analyze.py) first.")

    print(f"Refining poses for: {video.name}  (pass 1: {poses_path.name})")
    pass1 = load_pass1(poses_path)
    refined = refine_poses(str(video), pass1, mode=args.mode, device=args.device,
                           max_frames=args.max_frames)

    if args.replace_cache:
        backup = video.with_name(video.stem + "_poses_pass1.json")
        if not backup.exists():
            poses_path.rename(backup)
            print(f"  Pass-1 poses kept as: {backup.name}")
        out_path = video.with_name(video.stem + "_poses.json")
    else:
        out_path = Path(args.out) if args.out else video.with_name(video.stem + "_poses_refined.json")

    out_path.write_text(json.dumps(pe.poses_to_serialisable(refined)), encoding="utf-8")
    print(f"  Refined poses saved to: {out_path}")

    if args.preview is not None:
        save_preview(str(video), pass1, refined, args.preview,
                     str(video.with_name(video.stem + "_refine_preview.png")))


if __name__ == "__main__":
    main()
