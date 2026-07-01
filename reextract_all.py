#!/usr/bin/env python3
"""
Batch re-extraction: run the full current pipeline over every cached clip.

For each `<stem>_poses.json` under me/ and pros/:
  1. keep the original as `<stem>_poses_pass1.json` (skipped if already there)
  2. pass 2 — pose_refine (top-down RTMPose, Halpe-26 w/ feet)
  3. pass 3 — pose_lift  (VideoPose3D root-relative 3D)
  4. overwrite `<stem>_poses.json` with the refined + lifted result

Clips whose cache is already halpe26 are only re-lifted (refine is idempotent
per model, no need to redo it). Run whenever the pipeline changes so your
clips and the pro references stay on the same measurement scale.

Usage:
    python reextract_all.py [--dry-run]
"""

import argparse
import shutil
import time
from pathlib import Path

import pose_refine as pr
import pose_lift as pl

ROOT = Path(__file__).parent
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def find_video(poses_path: Path) -> Path | None:
    stem = poses_path.name[: -len("_poses.json")]
    for p in poses_path.parent.iterdir():
        if p.suffix.lower() in VIDEO_EXTS and p.stem == stem:
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Refine + lift every cached clip.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be done")
    ap.add_argument("--mode", default="balanced", choices=("lightweight", "balanced", "performance"))
    args = ap.parse_args()

    targets = []
    for d in ("me", "pros"):
        base = ROOT / d
        if base.is_dir():
            targets += sorted(base.rglob("*_poses.json"))

    print(f"{len(targets)} cached clips found\n")
    t_all = time.time()
    done = skipped = failed = 0

    for poses_path in targets:
        rel = poses_path.relative_to(ROOT)
        video = find_video(poses_path)
        if video is None:
            print(f"SKIP {rel} — no matching video file")
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD process {rel}  (video: {video.name})")
            continue

        print(f"=== {rel} ===")
        try:
            data = pr.load_pass1(poses_path)
            if data.get("keypoint_format") != "halpe26":
                backup = poses_path.with_name(poses_path.name[:-len(".json")] + "_pass1.json")
                if not backup.exists():
                    shutil.copy2(poses_path, backup)
                data = pr.refine_poses(str(video), data, mode=args.mode)
            else:
                print("  already halpe26 — refine skipped")
            data = pl.lift_poses(data)
            pl.save_poses(data, poses_path)
            print(f"  updated {rel}\n")
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print(f"Done: {done} updated, {skipped} skipped, {failed} failed "
          f"({(time.time()-t_all)/60:.1f} min)")


if __name__ == "__main__":
    main()
