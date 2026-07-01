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
# Pro reference entries: (video_path, poses_json_path, display_label, lead_id, couple)
# lead_id = which tracked Dancer ID (1 or 2) is the pro LEAD in that clip.
# couple  = which couple this clip belongs to. The gap analysis groups clips by
#           couple, averages each couple's clips together, and shows a SEPARATE gap
#           section per couple (rather than one pooled pro average). Use the same
#           `couple` string across a couple's clips to pool them.
# NOTE: Dancer IDs are re-assigned on every extraction, and lead_id is PER CLIP (the
# same pro can be Dancer 1 in one clip and Dancer 2 in another). Re-verify whenever
# pro poses are regenerated (identify the lead from a clear MID-PERFORMANCE frame —
# the first frames are often an intro/title card; see SKILL.md).
#
# Configuration priority:
#   1. pro_refs.json in DANCE_DIR  — explicit list [{video, poses, label, lead_id, couple}]
#   2. Auto-discover pros/*/       — each subfolder = one couple, ALL its clips with a
#      matching *_poses.json (lead_id defaults to 1; a reminder is printed to verify)
#
# Create a pro_refs.json from the provided pro_refs.example.json to set these correctly.

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
            result.append((vp, pp, e["label"], int(e.get("lead_id", 1)),
                           e.get("couple", e["label"])))
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
            couple = sub.name           # one folder = one couple; all its clips group together
            found = False
            for vp in videos:
                pp_candidates = sorted(sub.glob(f"{vp.stem}_poses.json"))
                if not pp_candidates:
                    continue
                found = True
                result.append((vp, pp_candidates[0], vp.stem, 1, couple))
            if found:
                print(f"  [pro-refs] Auto-discovered couple '{couple}' — lead_id defaulting to 1 "
                      "for its clips. Create pro_refs.json to set the correct per-clip lead_id.")
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


def _load_or_extract(video_path: pathlib.Path, model_name: str = "yolov8m-pose.pt",
                     seed: dict | None = None) -> dict:
    """Load cached poses if available, otherwise run extraction (slow).

    `seed` (crowd mode) = {"frame_idx": int, "points": [(x,y), (x,y)]} → always
    re-extracts with the seeded matcher and overwrites the cache.
    """
    poses_path = _poses_path_for(video_path)
    if seed is None and poses_path.exists():
        print(f"  Loading cached poses from {poses_path.name} …")
        poses = json.loads(poses_path.read_text(encoding="utf-8"))
        cached_model = poses.get("model")
        if cached_model and cached_model != model_name:
            print(f"  NOTE: cached poses were extracted with '{cached_model}', not "
                  f"'{model_name}'. Delete {poses_path.name} to re-extract.")
    else:
        why = "seeded re-extraction" if seed is not None else "pose extraction"
        print(f"  Running {why} on {video_path.name} with {model_name} …")
        print("  (this takes a few minutes — result cached as "
              f"{poses_path.name} for next time)")
        poses = pe.extract_poses(
            str(video_path), model_name=model_name,
            seed_frame_idx=(seed or {}).get("frame_idx"),
            seed_points=(seed or {}).get("points"),
        )
        poses_path.write_text(json.dumps(poses, default=str), encoding="utf-8")
        print(f"  Poses saved → {poses_path.name}")
    return _normalise_poses(poses)


def _seed_preview(video_path: pathlib.Path, t_sec: float, model_name: str):
    """Crowd mode step 1: detect everyone in one frame, save a numbered preview image
    and a sidecar JSON of detection centres, so the user can point out the couple."""
    import cv2
    frame_idx, img, dets = pe.detect_single_frame(str(video_path), t_sec, model_name=model_name)
    seed_json = {"frame_idx": int(frame_idx), "t_sec": float(t_sec), "dets": []}
    for k, d in enumerate(dets):
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        col = (0, 200, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
        cv2.rectangle(img, (x0, max(0, y0 - 36)), (x0 + 96, max(0, y0 - 36) + 36), col, -1)
        cv2.putText(img, f"#{k}", (x0 + 6, max(0, y0 - 36) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2)
        seed_json["dets"].append({"idx": k,
                                  "center": [float(d["center"][0]), float(d["center"][1])],
                                  "conf": round(float(d["conf"]), 3)})
    png = video_path.with_name(video_path.stem + "_seed.png")
    js  = video_path.with_name(video_path.stem + "_seed.json")
    cv2.imwrite(str(png), img)
    js.write_text(json.dumps(seed_json, indent=2), encoding="utf-8")
    print(f"  Seed preview → {png.name}  ({len(dets)} people detected at {t_sec:.1f}s)")
    for d in seed_json["dets"]:
        print(f"    #{d['idx']}: center=({d['center'][0]:.0f},{d['center'][1]:.0f})  conf={d['conf']}")
    print("  Then re-run with: --seed-me-idx <n> --seed-partner-idx <n>")


def _load_seed(video_path: pathlib.Path, me_idx: int, partner_idx: int) -> dict:
    """Crowd mode step 2: read the seed sidecar and build the seed for extraction
    (dancer 1 = you, dancer 2 = partner)."""
    js = video_path.with_name(video_path.stem + "_seed.json")
    if not js.exists():
        sys.exit("ERROR: no seed preview found — run with --seed-frame <seconds> first.")
    sj = json.loads(js.read_text(encoding="utf-8"))
    by_idx = {d["idx"]: d["center"] for d in sj["dets"]}
    if me_idx not in by_idx or partner_idx not in by_idx:
        sys.exit(f"ERROR: seed indices not found; available: {sorted(by_idx)}")
    return {"frame_idx": sj["frame_idx"],
            "points": [tuple(by_idx[me_idx]), tuple(by_idx[partner_idx])]}


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

def _orient_lead_first(poses: dict) -> None:
    """Swap tracked Dancer 1 <-> 2 in place so the actual LEAD becomes Dancer 1.

    The tracker numbers the two dancers arbitrarily. Once we know which tracked dancer
    is the lead (from --me/--me-id + --role), relabelling so the lead is Dancer 1 makes
    every positional 'lead'/'follow' metric and report section reflect the TRUE roles
    rather than tracker order. Keys are ints here (poses already normalised).
    """
    for f in poses.get("frames", []):
        d = f.get("dancers")
        if not d:
            continue
        f["dancers"] = {(2 if k == 1 else 1 if k == 2 else k): v for k, v in d.items()}
    if poses.get("dancer_ids"):
        poses["dancer_ids"] = sorted(int(i) for i in poses["dancer_ids"])


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
                include_partner: bool = False, spotlight: bool = False,
                my_role: str = "lead") -> str:
    """Build a concise gap-comparison table, broken out PER COUPLE.

    `you_id` is the tracked Dancer ID that is YOU in this video, and `my_role`
    ("lead"/"follow") is the role you dance. Clips are grouped by couple; each couple
    gets its own section with its clips averaged together. Role-specific rows compare
    YOU against that couple's dancer of the SAME role. When `include_partner` is set,
    the same metrics are also shown for your PARTNER (the other role) vs that couple's
    other-role dancer. Partnership rows are role-agnostic.

    pro_entries items are (label, metrics, lead_id, couple) where lead_id is which
    Dancer ID is that clip's lead and couple is the grouping key.
    """

    partner_role             = "follow" if my_role == "lead" else "lead"
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

    def _pro_avg(entries, category, subkey, kind, role):
        # Average one couple's clips, comparing against the pro of the SAME role as
        # whoever this row is about. `entries` are this couple's (label, metrics, lead_id).
        want_role = my_role if role == "you" else partner_role
        vals = []
        for _lbl, pm, lead_id in entries:
            follow_id = 2 if lead_id == 1 else 1
            target_id = lead_id if want_role == "lead" else follow_id
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
        ("1-foot airborne % (true single-leg)",      "leg_action",  "one_foot_airborne_pct",            "side", True,  ".1f"),
        ("Ball-of-foot % (rolling action)",          "leg_action",  "ball_foot_pct",                    "side", True,  ".1f"),
        ("Toe-first landings % (roll thru foot)",    "leg_action",  "art_toe_first_pct",                "side", True,  ".1f"),
        ("Weight-only traveling (lower=better)",     "leg_action",  "step_count_weight_only_traveling", "side", False, ".0f"),
        ("Articulated traveling",                    "leg_action",  "step_count_articulated_traveling", "side", True,  ".0f"),
        ("Slotted movement range (BH)",              "travel",      "slot_travel_range_bh",             "side", True,  ".3f"),
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
        ("Torso pitch range (deg)",                  "body_action", "pitch_range_deg",                  "side", True,  ".1f"),
        ("Upper/lower rotation dissoc (deg)",        "body_action", "upper_lower_rotation_mean_deg",    "side", True,  ".1f"),
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
        ("Partner distance variance",          "weight_countering", "partner_distance_std",   True, ".3f"),
        ("Posts detected",                     "weight_countering", "post_count",             True, ".0f"),
        ("Stretch-leading posts",              "weight_countering", "post_stretch_leading",   True, ".0f"),
        ("Compression-leading posts",          "weight_countering", "post_compression_leading",True, ".0f"),
        ("Stretch range after post (BH)",      "weight_countering", "post_max_stretch_mean",  True, ".3f"),
        ("Floor travel range (BH)",            "travel",            "couple_travel_range_bh", True, ".3f"),
        ("Accent coverage % (either)",         "musicality",        "accent_covered_pct",     True, ".1f"),
    ]

    roles = ["you", "partner"] if include_partner else ["you"]

    # Group clips by couple (insertion order preserved), so each couple gets its own
    # section with its clips averaged together — rather than one pooled pro average.
    groups = {}
    for lbl, pm, lead_id, couple in pro_entries:
        groups.setdefault(couple, []).append((lbl, pm, lead_id))

    header = f"  you = Dancer {you_id} ({my_role}); rows compare you vs each couple's {my_role.upper()}"
    if include_partner:
        header += (f"\n  partner = the {partner_role}; partner rows compare vs each "
                   f"couple's {partner_role.upper()}")
    lines = [
        "",
        "=" * 72,
        "  GAP ANALYSIS vs PRO REFERENCES  (broken out per couple)",
        header,
        "=" * 72,
    ]

    def _emit(label, am_val, pa, hib, fmt, suffix, vlabel, note=""):
        if am_val is None or pa is None:
            return
        delta = am_val - pa
        arrow = "▲" if (delta > 0) == hib else "▼"
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  {label + suffix:<46s}  {vlabel}={am_val:{fmt}}  pro avg={pa:{fmt}}  "
            f"{arrow} {sign}{delta:{fmt}}{note}"
        )

    for couple, entries in groups.items():
        clips = ", ".join(lbl for lbl, _pm, _lid in entries)
        n = len(entries)
        lines += [
            "",
            "-" * 72,
            f"  vs {couple}  —  averaged over {n} clip{'s' if n != 1 else ''}: {clips}",
            "-" * 72,
        ]
        for role in roles:
            if include_partner:
                disp = my_role if role == "you" else partner_role
                lines.append(f"  -- {role.upper()} ({disp.upper()}) --")
            for label, category, subkey, kind, hib, fmt in role_checks:
                _emit(label, _am(category, subkey, kind, role),
                      _pro_avg(entries, category, subkey, kind, role),
                      hib, fmt, f" — {role}", role)

        if include_partner:
            lines.append("  -- PARTNERSHIP (both) --")
        for label, category, subkey, hib, fmt in pair_checks:
            note = ""
            if subkey == "couple_travel_range_bh" and not spotlight:
                note = "   (not spotlight — lower expected)"
            _emit(label, _am(category, subkey, "pair", "you"),
                  _pro_avg(entries, category, subkey, "pair", "you"),
                  hib, fmt, "", "you", note=note)

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
                        help="Which side YOU start on: left=Dancer 1, "
                             "right=Dancer 2 (default left). Resolved to a Dancer ID from "
                             "the first clean frame. Controls which tracked dancer the "
                             "'you' rows and report headers refer to. Pair with --role to "
                             "say whether you lead or follow.")
    parser.add_argument("--role", choices=["lead", "follow"], default="lead",
                        help="Your dance role (default lead). 'you' rows are compared "
                             "against each pro of the SAME role; with --partner, your "
                             "partner is compared against each pro of the other role.")
    parser.add_argument("--me-id", type=int, choices=[1, 2], default=None,
                        help="Directly set which tracked Dancer ID is you, overriding --me. "
                             "Use when auto side-detection picks the wrong dancer (entry-heavy "
                             "clips): view a labelled frame, then pass 1 or 2.")
    parser.add_argument("--partner", action="store_true",
                        help="Also show your PARTNER's comparison (partner vs each pro of "
                             "the other role) in the gap analysis, not just your own.")
    parser.add_argument("--spotlight", action="store_true",
                        help="Mark this clip as a spotlight (full-floor showcase). When set, "
                             "couple-around-the-room travel is compared to the pro baseline "
                             "normally. Otherwise the clip is treated as contained (prelim/"
                             "practice) and that one gap row is annotated 'lower expected'.")
    parser.add_argument("--pose-model", default="yolov8m-pose.pt",
                        help="YOLOv8 pose model for extraction (n/s/m/l/x; default m). "
                             "Use l or x for max accuracy on small or crowded figures "
                             "(slower; poses are cached once).")
    parser.add_argument("--seed-frame", type=float, default=None,
                        help="CROWD MODE step 1: seconds into the video to grab a frame; "
                             "saves a numbered preview (<stem>_seed.png) of everyone detected, "
                             "then exits so you can pick your couple.")
    parser.add_argument("--seed-me-idx", type=int, default=None,
                        help="CROWD MODE step 2: the preview number (#) that is YOU.")
    parser.add_argument("--seed-partner-idx", type=int, default=None,
                        help="CROWD MODE step 2: the preview number (#) that is your PARTNER. "
                             "Re-extracts tracking only your couple out of the crowd.")
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

    # ── crowd mode: seed the target couple ───────────────────────────────────
    seeded = args.seed_me_idx is not None and args.seed_partner_idx is not None
    if args.seed_frame is not None and not seeded:
        # Step 1: render the numbered preview and exit so the user can pick.
        print(f"\nSeeding: {video_path.name}")
        _seed_preview(video_path, args.seed_frame, args.pose_model)
        return
    seed = _load_seed(video_path, args.seed_me_idx, args.seed_partner_idx) if seeded else None

    # ── step 2: pose extraction (cached) ────────────────────────────────────
    print(f"\nAnalysing: {video_path.name}")
    poses = _load_or_extract(video_path, model_name=args.pose_model, seed=seed)
    poses["video_path"] = str(video_path)

    # Resolve which tracked Dancer ID is the user (their role is set by --role).
    if seeded:
        you_id = 1   # seed step 2 pins dancer 1 = you, dancer 2 = partner
        print(f"  You ({args.role}) = Dancer 1 (seeded as the person you picked, #{args.seed_me_idx})")
    elif args.me_id is not None:
        you_id = args.me_id
        print(f"  You ({args.role}) = Dancer {you_id} (set explicitly via --me-id)")
    else:
        you_id = _dancer_on_side(poses, args.me)
        print(f"  You ({args.role}) start on the {args.me} → Dancer {you_id}")

    # Orient tracking so the actual LEAD is Dancer 1, making every positional
    # 'lead'/'follow' metric and report section reflect true roles (not tracker order).
    partner_tracked = 2 if you_id == 1 else 1
    lead_tracked    = you_id if args.role == "lead" else partner_tracked
    if lead_tracked != 1:
        _orient_lead_first(poses)
        print("  (Oriented tracking so the lead is Dancer 1 — metrics reflect true roles.)")
    you_id = 1 if args.role == "lead" else 2

    # ── step 3: metrics ──────────────────────────────────────────────────────
    print("  Computing metrics …")
    metrics = dm.compute_all_metrics(poses)
    metrics["spotlight"] = bool(args.spotlight)

    # ── step 4: report ───────────────────────────────────────────────────────
    print("  Building report …\n")
    report = dr.build_report(str(video_path), poses, metrics, you_id=you_id,
                             me=(None if args.me_id is not None else args.me),
                             spotlight=args.spotlight, my_role=args.role)

    report_path = out_dir / (video_path.stem + "_report.txt")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n  Report saved → {report_path}")

    # ── step 5: pro comparison (optional) ───────────────────────────────────
    if args.compare_pros:
        pro_entries = []
        for vp, pp, lbl, *rest in PRO_REFS:
            lead_id = rest[0] if len(rest) > 0 else 1
            couple  = rest[1] if len(rest) > 1 else lbl
            pm = _load_pro_metrics(vp, pp)
            if pm:
                pro_entries.append((lbl, pm, lead_id, couple))

        if pro_entries:
            gap = _gap_report(metrics, pro_entries, you_id=you_id,
                              include_partner=args.partner, spotlight=args.spotlight,
                              my_role=args.role)
            print(gap)
            gap_path = out_dir / (video_path.stem + "_gap_analysis.txt")
            gap_path.write_text(gap, encoding="utf-8")
            print(f"  Gap analysis saved → {gap_path}")
        else:
            print("  (No pro reference poses found — skipping comparison)")


if __name__ == "__main__":
    main()
