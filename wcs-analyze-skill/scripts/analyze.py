"""
WCS Video Analysis Runner
-------------------------
Usage:
    python analyze.py <video_path_or_youtube_url> [--compare-pros] [--output-dir <dir>]

Handles the full pipeline:
  1. Download from YouTube if a URL is given
  2. Load or run pose extraction (cached as <stem>_poses.json next to the video)
  3. Compute metrics via dance_metrics.py
  4. Generate and print the report via dance_review.py
  5. Optionally print a gap comparison against the pro reference videos
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

# ── locate the Dance Analysis project ───────────────────────────────────────
# Use DANCE_ANALYSIS_DIR env var if set; otherwise fall back to cwd.
import os as _os
DANCE_DIR = pathlib.Path(_os.environ.get("DANCE_ANALYSIS_DIR", ".")).resolve()
sys.path.insert(0, str(DANCE_DIR))

import dance_metrics as dm
import dance_review as dr
import pose_extraction as pe

# ── pro reference configuration ──────────────────────────────────────────────
# Pro reference entries: (video_path, poses_json_path, display_label, lead_id)
# lead_id = which tracked Dancer ID (1 or 2) is the pro LEAD in that clip.
# NOTE: Dancer IDs are re-assigned on every extraction — re-verify lead_id whenever
# pro poses are regenerated (identify the lead from a clear open-position frame; see
# "Adding a new pro reference video" in SKILL.md).
#
# Configuration priority:
#   1. pro_refs.json in DANCE_DIR  — explicit list [{video, poses, label, lead_id}]
#   2. Auto-discover pros/*/       — first video + matching *_poses.json per subfolder
#      (lead_id defaults to 1; a reminder is printed to verify this)
#
# Create a pro_refs.json from the provided pro_refs.example.json to set lead_id correctly.

def _load_pro_refs(dance_dir: pathlib.Path) -> list:
    """Return a list of (video_path, poses_path, label, lead_id) tuples."""
    json_cfg = dance_dir / "pro_refs.json"
    if json_cfg.exists():
        entries = json.loads(json_cfg.read_text(encoding="utf-8"))
        result = []
        for e in entries:
            vp = pathlib.Path(e["video"])
            pp = pathlib.Path(e["poses"])
            if not vp.is_absolute():
                vp = dance_dir / vp
            if not pp.is_absolute():
                pp = dance_dir / pp
            result.append((vp, pp, e["label"], int(e.get("lead_id", 1))))
        return result

    # Auto-discover pros/*/
    pros_dir = dance_dir / "pros"
    result = []
    if pros_dir.is_dir():
        for sub in sorted(pros_dir.iterdir()):
            if not sub.is_dir():
                continue
            videos = sorted(sub.glob("*.mp4")) + sorted(sub.glob("*.mov")) + sorted(sub.glob("*.MOV"))
            if not videos:
                continue
            vp = videos[0]
            pp_candidates = sorted(sub.glob(f"{vp.stem}_poses.json"))
            if not pp_candidates:
                continue
            pp = pp_candidates[0]
            label = sub.name
            print(f"  [pro-refs] Auto-discovered: {label} — lead_id defaulting to 1. "
                  "Create pro_refs.json to set the correct lead_id.")
            result.append((vp, pp, label, 1))
    return result


PRO_REFS = _load_pro_refs(DANCE_DIR)

# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_kps(raw):
    """Convert keypoints from JSON (list, numpy-string repr, or ndarray) to ndarray."""
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, str):
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        return np.array(nums, dtype=float).reshape(-1, 3)
    arr = np.array(raw, dtype=float)
    return arr.reshape(-1, 3) if arr.ndim == 1 else arr


def _normalise_poses(poses: dict) -> dict:
    """Fix JSON round-trip issues: string dancer keys → int, kps → ndarray."""
    for f in poses["frames"]:
        d = f.get("dancers", {})
        if not d:
            continue
        if isinstance(next(iter(d)), str):
            f["dancers"] = {int(k): v for k, v in d.items()}
        for did, kps in list(f["dancers"].items()):
            if not isinstance(kps, np.ndarray):
                f["dancers"][did] = _parse_kps(kps)
    return poses


def _poses_path_for(video_path: pathlib.Path) -> pathlib.Path:
    """Return the expected poses JSON path alongside the video file."""
    return video_path.with_name(video_path.stem + "_poses.json")


def _load_or_extract(video_path: pathlib.Path) -> dict:
    """Load cached poses if available, otherwise run extraction (slow)."""
    poses_path = _poses_path_for(video_path)
    if poses_path.exists():
        print(f"  Loading cached poses from {poses_path.name} …")
        poses = json.loads(poses_path.read_text(encoding="utf-8"))
    else:
        print(f"  Running pose extraction on {video_path.name} …")
        print("  (this takes a few minutes — result cached as "
              f"{poses_path.name} for next time)")
        poses = pe.extract_poses(str(video_path))
        poses_path.write_text(json.dumps(poses, default=str), encoding="utf-8")
        print(f"  Poses saved → {poses_path.name}")
    return _normalise_poses(poses)


def _download_youtube(url: str, out_dir: pathlib.Path) -> pathlib.Path:
    """Download a YouTube video+audio and merge into a single mp4."""
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = str(pathlib.Path(ffmpeg_exe).parent)

    vid_id = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    stem = vid_id.group(1) if vid_id else "yt_video"
    mp4_path = out_dir / f"{stem}.mp4"

    if mp4_path.exists():
        print(f"  Already downloaded: {mp4_path.name}")
        return mp4_path

    print(f"  Downloading {url} …")
    r = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--ffmpeg-location", ffmpeg_dir,
            "-o", str(mp4_path),
            url,
        ],
        capture_output=True,
        text=True,
    )

    if not mp4_path.exists():
        # yt-dlp may have downloaded streams separately — find and merge
        video_part = next(out_dir.glob(f"{stem}.f*.mp4"), None)
        audio_part = next(out_dir.glob(f"{stem}.f*.m4a"), None)
        if video_part and audio_part:
            print("  Merging video + audio streams …")
            subprocess.run(
                [ffmpeg_exe, "-i", str(video_part), "-i", str(audio_part),
                 "-c:v", "copy", "-c:a", "aac", "-y", str(mp4_path)],
                check=True, capture_output=True,
            )
            video_part.unlink(missing_ok=True)
            audio_part.unlink(missing_ok=True)
        else:
            print(r.stderr[-2000:])
            raise RuntimeError("Download failed and no separate streams found.")

    print(f"  Saved → {mp4_path.name}")
    return mp4_path


# ── role resolution ───────────────────────────────────────────────────────────

def _dancer_on_side(poses: dict, side: str) -> int:
    """Return the Dancer ID (1 or 2) on the given side ("left"/"right") near the start
    of the dancing.

    The tracker's Dancer 1 is NOT guaranteed to be the left dancer (it's whoever was
    detected first), so we resolve the user's stated starting side to an actual ID by
    looking at real positions. We deliberately skip the very first frames: early on the
    floor is often empty / mid-walk-on, and reflections or bystanders get detected, so
    the first two-person frame is unreliable. Instead we use the earliest frame where
    BOTH dancers are at near-full size (≥ 70% of each one's median body height), i.e.
    both are actually on the floor dancing.
    """
    frames = poses.get("frames", [])

    bhs = {1: [], 2: []}
    for f in frames:
        d = f.get("dancers", {})
        for did in (1, 2):
            if did in d:
                h = pe.body_height(d[did])
                if h > 10:
                    bhs[did].append(h)
    med = {did: (float(np.median(v)) if v else 0.0) for did, v in bhs.items()}

    def _resolve(d):
        x1 = float(pe.get_center(d[1])[0])
        x2 = float(pe.get_center(d[2])[0])
        left_id = 1 if x1 <= x2 else 2
        return left_id if side == "left" else (2 if left_id == 1 else 1)

    # Preferred: first frame where both dancers are near full size (truly on the floor)
    if med[1] > 0 and med[2] > 0:
        for f in frames:
            d = f.get("dancers", {})
            if 1 in d and 2 in d \
               and pe.body_height(d[1]) >= 0.7 * med[1] \
               and pe.body_height(d[2]) >= 0.7 * med[2]:
                return _resolve(d)

    # Fallback: first two-person frame at all
    for f in frames:
        d = f.get("dancers", {})
        if 1 in d and 2 in d:
            return _resolve(d)

    return 1 if side == "left" else 2   # last resort: assume left == Dancer 1


# ── pro gap comparison ────────────────────────────────────────────────────────

def _load_pro_metrics(video_path: pathlib.Path, poses_path: pathlib.Path) -> dict | None:
    if not poses_path.exists():
        return None
    poses = _normalise_poses(json.loads(poses_path.read_text(encoding="utf-8")))
    poses["video_path"] = str(video_path)
    return dm.compute_all_metrics(poses)


def _cols_for(dancer_id: int) -> tuple:
    """Map a tracked Dancer ID to its metric column keys: ('lead','a') or ('follow','b').

    compute_all_metrics labels Dancer 1 → 'lead'/'a' and Dancer 2 → 'follow'/'b'.
    """
    return ("lead", "a") if dancer_id == 1 else ("follow", "b")


def _gap_report(am_metrics: dict, pro_entries: list, you_id: int = 1,
                include_partner: bool = False) -> str:
    """Build a concise gap-comparison table (you vs pro average).

    `you_id` is the tracked Dancer ID that is YOU (the lead) in this video. Role-
    specific rows compare YOU against each pro's LEAD. When `include_partner` is set,
    the same metrics are also shown for your PARTNER (the follow) vs each pro's FOLLOW
    — i.e. a like-for-like analysis of the follower. Partnership rows are role-agnostic.

    pro_entries items are (label, metrics, lead_id) where lead_id is which Dancer ID
    is that pro's lead.
    """

    you_side, you_ab         = _cols_for(you_id)
    partner_id               = 2 if you_id == 1 else 1
    partner_side, partner_ab = _cols_for(partner_id)

    def _dig(m, category, subkey, kind, side, ab):
        """Pull one value, resolving how the metric encodes the dancer role."""
        if kind == "pair":                       # role-agnostic (partnership)
            node = m.get(category, {})
            v = node.get(subkey) if isinstance(node, dict) else None
        elif kind == "side":                     # leg_action_/body_action_<side>
            node = m.get(f"{category}_{side}", {})
            v = node.get(subkey) if isinstance(node, dict) else None
        else:                                     # "ab" — musicality, keyed _a / _b
            node = m.get("musicality", {})
            v = node.get(f"{subkey}_{ab}") if isinstance(node, dict) else None
        return float(v) if isinstance(v, (int, float)) else None

    def _am(category, subkey, kind, role):
        side = you_side if role == "you" else partner_side
        ab   = you_ab   if role == "you" else partner_ab
        return _dig(am_metrics, category, subkey, kind, side, ab)

    def _pro_avg(category, subkey, kind, role):
        vals = []
        for _lbl, pm, lead_id in pro_entries:
            follow_id = 2 if lead_id == 1 else 1
            target_id = lead_id if role == "you" else follow_id
            p_side, p_ab = _cols_for(target_id)
            v = _dig(pm, category, subkey, kind, p_side, p_ab)
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    # Role-specific metrics, emitted once per role — (label, category, subkey, kind, hib, fmt)
    role_checks = [
        ("Rise/fall typical (bounce on avg steps)",  "leg_action",  "rise_fall_typical",                "side", True,  ".4f"),
        ("Rise/fall dynamic (biggest level changes)","leg_action",  "rise_fall_dynamic",                "side", True,  ".3f"),
        ("1-foot balance %",                         "leg_action",  "one_foot_pct",                     "side", True,  ".1f"),
        ("Weight-only traveling (lower=better)",     "leg_action",  "step_count_weight_only_traveling", "side", False, ".0f"),
        ("Articulated traveling",                    "leg_action",  "step_count_articulated_traveling", "side", True,  ".0f"),
        ("Art. free-leg prep knee flex (deg)",       "leg_action",  "art_free_knee_flex_deg",           "side", True,  ".1f"),
        ("Art. free-leg prep hip flex (deg)",        "leg_action",  "art_free_hip_flex_deg",            "side", True,  ".1f"),
        ("Art. standing-leg knee flex med (deg)",    "leg_action",  "art_weighted_knee_flex_deg",       "side", True,  ".1f"),
        ("Art. standing-leg knee flex p90 (ceiling)","leg_action",  "art_weighted_knee_p90",            "side", True,  ".1f"),
        ("Art. free-leg knee flex p90 (ceiling)",    "leg_action",  "art_free_knee_p90",                "side", True,  ".1f"),
        ("Art. free knee-hip coordination",          "leg_action",  "art_knee_hip_coord",               "side", True,  ".2f"),
        ("Art. bend smoothness",                     "leg_action",  "art_smoothness",                   "side", True,  ".3f"),
        ("Art. straighten recovery %",               "leg_action",  "art_straighten_pct",               "side", True,  ".1f"),
        ("Art. prep→arrival sequencing %",           "leg_action",  "art_prep_pct",                     "side", True,  ".1f"),
        ("Motion smoothness",                        "body_action", "motion_smoothness",                "side", True,  ".3f"),
        ("Texture match (move vs song)",             "musicality",  "texture_match",                    "ab",   True,  ".3f"),
        ("Bounce match (beat rhythm)",               "musicality",  "bounce_match",                     "ab",   True,  ".3f"),
        ("On-beat articulated steps %",              "musicality",  "on_beat_pct",                      "ab",   True,  ".1f"),
        ("Timing consistency (ms, lower=better)",    "musicality",  "timing_ms",                        "ab",   False, ".0f"),
        ("Syncopation %",                            "musicality",  "syncopation_pct",                  "ab",   True,  ".1f"),
        ("Accent response % (any channel)",          "musicality",  "accent_response_pct",              "ab",   True,  ".1f"),
        ("Accent hit intensity",                     "musicality",  "accent_hit_mean",                  "ab",   True,  ".2f"),
    ]
    # Role-agnostic partnership metrics, emitted once — (label, category, subkey, hib, fmt)
    pair_checks = [
        ("Partner distance variance",  "weight_countering", "partner_distance_std", True, ".3f"),
        ("Posts detected",             "weight_countering", "post_count",           True, ".0f"),
        ("Accent coverage % (either)", "musicality",        "accent_covered_pct",   True, ".1f"),
    ]

    roles = ["you", "partner"] if include_partner else ["you"]

    header = f"  you = Dancer {you_id} (lead); rows compare you vs each pro's LEAD"
    if include_partner:
        header += "\n  partner = the follow; partner rows compare vs each pro's FOLLOW"
    lines = [
        "",
        "=" * 72,
        "  GAP ANALYSIS vs PRO REFERENCE",
        f"  (averaging over: {', '.join(lbl for lbl, _pm, _lid in pro_entries)})",
        header,
        "=" * 72,
    ]

    def _emit(label, am_val, pa, hib, fmt, suffix, vlabel):
        if am_val is None or pa is None:
            return
        delta = am_val - pa
        arrow = "▲" if (delta > 0) == hib else "▼"
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  {label + suffix:<46s}  {vlabel}={am_val:{fmt}}  pro avg={pa:{fmt}}  "
            f"{arrow} {sign}{delta:{fmt}}"
        )

    for role in roles:
        if include_partner:
            lines.append(f"  -- {role.upper()} --")
        for label, category, subkey, kind, hib, fmt in role_checks:
            _emit(label, _am(category, subkey, kind, role), _pro_avg(category, subkey, kind, role),
                  hib, fmt, f" — {role}", role)

    if include_partner:
        lines.append("  -- PARTNERSHIP (both) --")
    for label, category, subkey, hib, fmt in pair_checks:
        _emit(label, _am(category, subkey, "pair", "you"), _pro_avg(category, subkey, "pair", "you"),
              hib, fmt, "", "you")

    lines += ["", "=" * 72]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WCS video analysis")
    parser.add_argument("input", help="Video file path or YouTube URL")
    parser.add_argument("--compare-pros", action="store_true",
                        help="Show gap comparison against pro reference videos")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: same as video)")
    parser.add_argument("--me", choices=["left", "right"], default="left",
                        help="Which side YOU (the lead) start on: left=Dancer 1, "
                             "right=Dancer 2 (default left). Resolved to a Dancer ID from "
                             "the first clean frame. Controls which tracked dancer the "
                             "'you' rows and report headers refer to.")
    parser.add_argument("--me-id", type=int, choices=[1, 2], default=None,
                        help="Directly set which tracked Dancer ID is you, overriding --me. "
                             "Use when auto side-detection picks the wrong dancer (entry-heavy "
                             "clips): view a labelled frame, then pass 1 or 2.")
    parser.add_argument("--partner", action="store_true",
                        help="Also show the follower's comparison (your partner vs each pro's "
                             "follow) in the gap analysis, not just the lead's.")
    args = parser.parse_args()

    # ── step 1: resolve input to a local video file ──────────────────────────
    is_url = args.input.startswith("http://") or args.input.startswith("https://")
    if is_url:
        out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(".")
        video_path = _download_youtube(args.input, out_dir)
    else:
        video_path = pathlib.Path(args.input)
        if not video_path.exists():
            sys.exit(f"ERROR: file not found: {video_path}")

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else video_path.parent

    # ── step 2: pose extraction (cached) ────────────────────────────────────
    print(f"\nAnalysing: {video_path.name}")
    poses = _load_or_extract(video_path)
    poses["video_path"] = str(video_path)

    # Resolve which tracked Dancer ID is the user (the lead).
    if args.me_id is not None:
        you_id = args.me_id
        print(f"  You (lead) = Dancer {you_id} (set explicitly via --me-id)")
    else:
        you_id = _dancer_on_side(poses, args.me)
        print(f"  You (lead) start on the {args.me} → Dancer {you_id}")

    # ── step 3: metrics ──────────────────────────────────────────────────────
    print("  Computing metrics …")
    metrics = dm.compute_all_metrics(poses)

    # ── step 4: report ───────────────────────────────────────────────────────
    print("  Building report …\n")
    report = dr.build_report(str(video_path), poses, metrics, you_id=you_id,
                             me=(None if args.me_id is not None else args.me))

    report_path = out_dir / (video_path.stem + "_report.txt")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n  Report saved → {report_path}")

    # ── step 5: pro comparison (optional) ───────────────────────────────────
    if args.compare_pros:
        pro_entries = []
        for vp, pp, lbl, *rest in PRO_REFS:
            lead_id = rest[0] if rest else 1
            pm = _load_pro_metrics(vp, pp)
            if pm:
                pro_entries.append((lbl, pm, lead_id))

        if pro_entries:
            gap = _gap_report(metrics, pro_entries, you_id=you_id,
                              include_partner=args.partner)
            print(gap)
            gap_path = out_dir / (video_path.stem + "_gap_analysis.txt")
            gap_path.write_text(gap, encoding="utf-8")
            print(f"  Gap analysis saved → {gap_path}")
        else:
            print("  (No pro reference poses found — skipping comparison)")


if __name__ == "__main__":
    main()
