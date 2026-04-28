#!/usr/bin/env python3
"""
Dance Review
============
Full analysis pipeline for a West Coast Swing dance video.
Extracts pose data, computes metrics, and outputs a structured report.

Usage:
    python dance_review.py video.mp4
    python dance_review.py video.mp4 --out report.txt
    python dance_review.py video.mp4 --poses existing.poses.json
    python dance_review.py video.mp4 --skip 2 --conf 0.35 --out report.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pose_extraction import extract_poses, poses_to_serialisable, poses_from_serialisable
from dance_metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72
DIV = "-" * 72


def _pct_bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {pct:5.1f}%"


def _flag(val: float, lo: float, hi: float, invert: bool = False) -> str:
    if invert:
        if val > hi:  return "  [!]"
        if val > lo:  return "  [~]"
        return "  [ok]"
    else:
        if val < lo:  return "  [!]"
        if val < hi:  return "  [~]"
        return "  [ok]"


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header(video_path: str, pose_data: dict, metrics: dict, lines: list):
    fps    = pose_data["fps"]
    frames = pose_data["frame_count"]
    ids    = pose_data.get("dancer_ids", [])
    dur    = frames / fps if fps else 0
    cam    = metrics.get("camera_setup", {})

    lines += [
        SEP,
        "  WEST COAST SWING DANCE REVIEW",
        SEP,
        f"  Video     : {os.path.basename(video_path)}",
        f"  Duration  : {_fmt_time(dur)}  ({frames} frames @ {fps:.1f} fps)",
        f"  Dancers   : {len(ids)} tracked  (IDs: {ids})",
        f"  Lead ID   : {ids[0] if ids else '?'}    Follow ID: {ids[1] if len(ids) > 1 else '?'}",
        "",
    ]

    tq = metrics.get("tracking_quality", {})
    if tq:
        lines += ["  TRACKING QUALITY", DIV]
        for label, d in tq.items():
            pct  = d["coverage_pct"]
            flag = "  [ok]" if pct >= 60 else ("  [~]" if pct >= 30 else "  [!] LOW — metrics unreliable")
            lines += [f"    {label.title():<8}: {pct:5.1f}% of frames tracked  ({d['frames_tracked']}/{d['total_frames']}){flag}"]
        low = [l for l, d in tq.items() if d["coverage_pct"] < 30]
        if low:
            lines += [f"    Tip: re-run pose extraction with a larger model:"]
            lines += [f"         python pose_extraction.py <video> --model yolov8s-pose.pt --conf 0.25"]
        lines += [""]

    if cam and "error" not in cam:
        view   = cam.get("view_angle", "?")
        elev   = cam.get("camera_elevation", "?")
        axis   = cam.get("partnership_axis_deg", 0.0)
        ratio  = cam.get("size_ratio", 1.0)
        notes  = cam.get("notes", [])

        lines += [
            "  CAMERA SETUP",
            DIV,
            f"    View angle to slot  : {view}  (partnership axis {axis:+.1f}° from horizontal)",
            f"    Camera elevation    : {elev}",
            f"    Apparent size ratio : {ratio:.2f}  (lead / follow; 1.0 = equal depth)",
        ]
        for note in notes:
            lines += [f"    [~] {note}"]
        lines += [""]


def _leg_action_section(label: str, data: dict, coverage_pct: float, lines: list):
    lines += [
        SEP,
        f"  LEG ACTION — {label.upper()}",
        SEP,
        "",
    ]
    if coverage_pct < 60:
        lines += [f"    [!] Only {coverage_pct:.0f}% of frames tracked — values below are estimates only.", ""]

    total    = data.get("step_count_total", 0)
    art      = data.get("step_count_articulated", 0)
    wo       = data.get("step_count_weight_only", 0)
    art_pct  = data.get("articulated_pct", 0.0)
    spm      = data.get("steps_per_minute", 0.0)
    kf_mean  = data.get("knee_flex_mean", 0.0)
    kf_art   = data.get("knee_flex_at_articulated", 0.0)
    kf_max   = data.get("knee_flex_max", 0.0)
    rf_typ   = data.get("rise_fall_typical", 0.0)
    rf_dyn   = data.get("rise_fall_dynamic", 0.0)
    rf_hz    = data.get("rise_fall_rhythm_hz", 0.0)
    triple   = data.get("triple_step_count", 0)
    one_foot = data.get("one_foot_pct", 0.0)
    two_foot = data.get("two_foot_pct", 0.0)

    art_trav = data.get("step_count_articulated_traveling", 0)
    art_ip   = data.get("step_count_articulated_in_place", 0)
    wo_trav  = data.get("step_count_weight_only_traveling", 0)
    wo_ip    = data.get("step_count_weight_only_in_place", 0)

    def _pct(n, d):
        return (100.0 * n / d) if d > 0 else 0.0

    lines += [
        f"    Total weight changes    : {total}  ({spm:.0f}/min)",
        f"    Articulated             : {art}  ({art_pct:.0f}% of total){_flag(art_pct, 25, 45)}",
        f"    Weight-only             : {wo}  ({100-art_pct:.0f}% of total)",
        "",
        f"                                Traveling       In-place",
        f"      Articulated           :   {art_trav:3d} ({_pct(art_trav, art):4.0f}%)     {art_ip:3d} ({_pct(art_ip, art):4.0f}%)",
        f"      Weight-only           :   {wo_trav:3d} ({_pct(wo_trav, wo):4.0f}%)     {wo_ip:3d} ({_pct(wo_ip, wo):4.0f}%)",
        f"      (articulated = heel lifts clear of ground)",
        f"      (traveling   = stance-center shifts > 5% body-height across the event — a foot moved)",
        f"      (weight-only traveling should be rare at high levels)",
        "",
        f"    Knee flexion (norm)     : mean {kf_mean:.3f}  max {kf_max:.3f}{_flag(kf_mean, 0.2, 0.35)}",
        f"    Knee flex at art. steps : {kf_art:.3f}{_flag(kf_art, 0.22, 0.38)}",
        f"      (0 = straight leg, 0.5 = deep bend; articulated steps should show more flex)",
        "",
        f"    Rise/fall (typical)     : {rf_typ:.3f}  (median bounce on average steps)",
        f"    Rise/fall (dynamic)     : {rf_dyn:.3f}  (95th-pct — biggest level changes)",
        f"    Rise/fall frequency     : {rf_hz:.2f} Hz  ({rf_hz * 60:.1f} cycles/min)",
        "",
        f"    Triple step sequences   : {triple}",
        "",
        f"    Balance",
        f"      1-foot (ankle ≥5% bh) : {_pct_bar(one_foot)}{one_foot:5.1f}%",
        f"      2-foot (both grounded): {_pct_bar(two_foot)}{two_foot:5.1f}%",
        f"      (1-foot: one ankle elevated ≥5% body-height — includes heel raise, brush, toe-touch)",
        "",
    ]


def _body_action_section(label: str, data: dict, coverage_pct: float, lines: list):
    lines += [
        SEP,
        f"  BODY ACTION — {label.upper()}",
        SEP,
        "",
    ]
    if coverage_pct < 60:
        lines += [f"    [!] Only {coverage_pct:.0f}% of frames tracked — values below are estimates only.", ""]

    pitch_rng  = data.get("pitch_range_deg",     0.0)
    pitch_hz   = data.get("pitch_rhythm_hz",     0.0)
    hs_lag     = data.get("hip_shoulder_lag_ms",  0.0)
    sh_lag     = data.get("shoulder_head_lag_ms", 0.0)
    smoothness = data.get("motion_smoothness",    0.0)
    sh_tilt    = data.get("shoulder_tilt_range_deg", 0.0)
    hi_tilt    = data.get("hip_tilt_range_deg",      0.0)
    sway_hz    = data.get("sway_rhythm_hz",          0.0)
    sway_diss  = data.get("upper_lower_sway_dissoc", 0.0)

    lines += [
        "  PITCH  (forward/backward lean along the slot)",
        DIV,
        f"    Pitch range             : {pitch_rng:.1f}°{_flag(pitch_rng, 5, 15)}",
        f"      (torso lean angle in image plane; 0° = upright, higher = more forward/back usage)",
        f"    Pitch rhythm            : {pitch_hz:.2f} Hz  ({pitch_hz * 60:.1f} cycles/min)",
        "",
        "  FLUIDITY  (sequential propagation through the body)",
        DIV,
        f"    Hip → shoulder lag      : {hs_lag:+.0f} ms{_flag(hs_lag, 20, 80)}",
        f"    Shoulder → head lag     : {sh_lag:+.0f} ms{_flag(sh_lag, 10, 60)}",
        f"      (positive = segment above follows the one below — bottom-up wave)",
        f"      (near zero = block body; negative = top leads bottom)",
        f"    Motion smoothness       : {smoothness:.3f}{_flag(smoothness, 0.55, 0.75)}",
        f"      (fraction of velocity energy below 2 Hz; higher = smoother flow)",
        "",
        "  SWAY  (side-to-side body tilt)",
        DIV,
        f"    Shoulder tilt range     : {sh_tilt:.1f}°{_flag(sh_tilt, 5, 15)}",
        f"    Hip tilt range          : {hi_tilt:.1f}°{_flag(hi_tilt, 5, 15)}",
        f"    Upper/lower dissociation: {sway_diss:.1f}°{_flag(sway_diss, 3, 10)}",
        f"      (how much shoulder tilt differs from hip tilt)",
        f"    Sway rhythm             : {sway_hz:.2f} Hz  ({sway_hz * 60:.1f} cycles/min)",
        "",
    ]


def _weight_countering_section(data: dict, lines: list):
    lines += [
        SEP,
        "  WEIGHT & COUNTERING (PARTNERSHIP)",
        SEP,
        "",
    ]

    if "error" in data:
        lines += [f"    [!] {data['error']}", ""]
        return

    conn_pcts   = data.get("connection_type_pcts", {})
    dist_mean   = data.get("partner_distance_mean", 0.0)
    dist_std    = data.get("partner_distance_std", 0.0)
    stretch     = data.get("stretch_pct", 0.0)
    compress    = data.get("compression_pct", 0.0)
    neutral     = round(max(0.0, 100 - stretch - compress), 1)
    lean_a      = data.get("lean_toward_conn_a", 0.0)
    lean_b      = data.get("lean_toward_conn_b", 0.0)
    counter     = data.get("counter_balance_pct", 0.0)
    slot_deg    = data.get("slot_direction_deg", 0.0)
    post_count  = data.get("post_count", 0)
    post_str    = data.get("post_max_stretch_mean", 0.0)
    post_cmp    = data.get("post_max_compression_mean", 0.0)

    conn_str = "  ".join(f"{k}: {v}%" for k, v in sorted(conn_pcts.items(), key=lambda x: -x[1]))

    lines += [
        f"    Connection type         : {conn_str or 'none detected'}",
        "",
        f"    Partner distance (norm) : {dist_mean:.3f}  ± {dist_std:.3f}",
        f"      (normalised to mean body height; 1.0 ≈ one body height apart)",
        "",
        f"    Stretch                 : {_pct_bar(stretch)}",
        f"    Compression             : {_pct_bar(compress)}",
        f"    Neutral / settled       : {_pct_bar(neutral)}",
        "",
        f"    Lead lean toward conn   : {lean_a:.1f}°{_flag(lean_a, 0, 45, invert=True)}",
        f"    Follow lean toward conn : {lean_b:.1f}°{_flag(lean_b, 0, 45, invert=True)}",
        f"    Counter-balance         : {_pct_bar(counter)}{_flag(counter, 20, 50)}",
        f"      (both dancers leaning into each other for shared resistance)",
        "",
        f"    Slot direction          : {slot_deg:.1f}° from horizontal",
        "",
        f"  POST DYNAMICS",
        DIV,
        f"    Posts detected          : {post_count}",
        f"      (connection point stationary ≥ 0.18 s — anchor for stretch/compression)",
    ]
    if post_count > 0:
        lines += [
            f"    Peak stretch (mean)     : {post_str:.3f} BH{_flag(post_str, 0.05, 0.15)}",
            f"      — how far dancers moved apart after the post, in body heights",
            f"    Peak compression (mean) : {post_cmp:.3f} BH",
            f"      — how much partners closed in after the post, in body heights",
        ]
    lines += [""]


def _musicality_section(data: dict, lines: list):
    lines += [
        SEP,
        "  MUSICALITY & TIMING",
        SEP,
        "",
    ]

    if "error" in data:
        lines += [f"    [!] {data['error']}", ""]
        return

    tempo      = data.get("tempo_bpm",         0.0)
    beats      = data.get("beat_count",         0)
    six_ct     = data.get("six_count_patterns", 0)
    eight_ct   = data.get("eight_count_patterns", 0)
    phrase_ct  = data.get("phrase_count",       0)

    def v(key):
        return data.get(key, 0.0)

    lines += [
        f"    Detected tempo          : {tempo:.1f} BPM  ({beats} beats)",
        "",
        "  FREE ARM STYLING",
        DIV,
        f"    Shoulder→wrist lag",
        f"      Lead                  : {v('arm_lag_a'):+.0f} ms{_flag(v('arm_lag_a'), 20, 80)}",
        f"      Follow                : {v('arm_lag_b'):+.0f} ms{_flag(v('arm_lag_b'), 20, 80)}",
        f"      (+ = wrist follows shoulder = fluid wave; ≈0 = block arm; − = wrist leads)",
        f"    Body–arm correlation",
        f"      Lead                  : {v('arm_body_corr_a'):+.3f}{_flag(v('arm_body_corr_a'), 0.3, 0.6)}",
        f"      Follow                : {v('arm_body_corr_b'):+.3f}{_flag(v('arm_body_corr_b'), 0.3, 0.6)}",
        f"      (Pearson r; higher = arms amplify/respond to body movement)",
        f"    Mean wrist speed (BH/s)",
        f"      Lead                  : {v('wrist_speed_a'):.3f}",
        f"      Follow                : {v('wrist_speed_b'):.3f}",
        "",
        "  TEXTURE",
        DIV,
        f"    Bounce match (beat rhythm)",
        f"      Lead                  : {_pct_bar(v('bounce_match_a') * 100)}{_flag(v('bounce_match_a'), 0.4, 0.7)}",
        f"      Follow                : {_pct_bar(v('bounce_match_b') * 100)}{_flag(v('bounce_match_b'), 0.4, 0.7)}",
        f"      (how well rise/fall rhythm aligns with beat; checks ×0.5, ×1, ×2 beat freq)",
        f"    Music–movement tracking",
        f"      Lead                  : {v('music_move_corr_a'):+.3f}{_flag(abs(v('music_move_corr_a')), 0.2, 0.5)}",
        f"      Follow                : {v('music_move_corr_b'):+.3f}{_flag(abs(v('music_move_corr_b')), 0.2, 0.5)}",
        f"      (Pearson r audio RMS vs movement speed; ±both indicate responsiveness to music energy)",
        f"    On-beat articulated steps",
        f"      Lead                  : {_pct_bar(v('on_beat_pct_a'))}{_flag(v('on_beat_pct_a'), 55, 80)}",
        f"      Follow                : {_pct_bar(v('on_beat_pct_b'))}{_flag(v('on_beat_pct_b'), 55, 80)}",
        f"    Timing consistency",
        f"      Lead                  : ±{v('timing_ms_a'):.0f} ms{_flag(v('timing_ms_a'), 0, 80, invert=True)}",
        f"      Follow                : ±{v('timing_ms_b'):.0f} ms{_flag(v('timing_ms_b'), 0, 80, invert=True)}",
        f"    Syncopation (weight-only & counts)",
        f"      Lead                  : {_pct_bar(v('syncopation_pct_a'))}",
        f"      Follow                : {_pct_bar(v('syncopation_pct_b'))}",
        "",
        "  PHRASE CHANGES",
        DIV,
        f"    Phrase boundaries       : {phrase_ct}  (every 32 beats = 8 bars of 4/4)",
        f"    Arm activity at phrase  ",
        f"      Lead                  : {_pct_bar(v('phrase_rsp_pct_a'))}{_flag(v('phrase_rsp_pct_a'), 30, 60)}",
        f"      Follow                : {_pct_bar(v('phrase_rsp_pct_b'))}{_flag(v('phrase_rsp_pct_b'), 30, 60)}",
        f"      (% of phrase changes where wrist speed ≥ 1.5× clip baseline — likely a hit or styling moment)",
        f"    Phrase hit intensity",
        f"      Lead                  : {v('phrase_hit_mean_a'):.2f}×  baseline wrist speed",
        f"      Follow                : {v('phrase_hit_mean_b'):.2f}×  baseline wrist speed",
        "",
        f"    WCS pattern fingerprint",
        f"      6-count sequences     : {six_ct}",
        f"      8-count sequences     : {eight_ct}",
        "",
    ]


def _flags_section(metrics: dict, lines: list):
    """Highlight the most notable issues and strengths across both dancers."""
    flags = []

    for label in ("lead", "follow"):
        la = metrics.get(f"leg_action_{label}", {})
        ba = metrics.get(f"body_action_{label}", {})

        if la.get("knee_flex_mean", 1) < 0.18:
            flags.append(f"  [!] {label.title()} — low knee flexion ({la['knee_flex_mean']:.3f}): consider deeper compression")
        if la.get("rise_fall_range", 1) < 0.02:
            flags.append(f"  [!] {label.title()} — minimal rise/fall: body may be flat/static")
        if la.get("triple_step_count", 99) < 3:
            flags.append(f"  [~] {label.title()} — few triple steps detected ({la.get('triple_step_count', 0)})")
        if ba.get("pitch_range_deg", 99) < 5:
            flags.append(f"  [~] {label.title()} — minimal pitch usage ({ba.get('pitch_range_deg', 0):.1f}°): body stays upright, little forward/back engagement")
        if ba.get("hip_shoulder_lag_ms", 0) < -30:
            flags.append(f"  [!] {label.title()} — negative hip→shoulder lag ({ba.get('hip_shoulder_lag_ms', 0):+.0f} ms): upper body leads lower, movement not grounded")
        if ba.get("motion_smoothness", 1) < 0.5:
            flags.append(f"  [~] {label.title()} — low motion smoothness ({ba.get('motion_smoothness', 0):.3f}): movement may be choppy or staccato")
        if ba.get("shoulder_tilt_range_deg", 99) < 5:
            flags.append(f"  [~] {label.title()} — minimal sway ({ba.get('shoulder_tilt_range_deg', 0):.1f}°): little upper-body tilt usage")

    wc = metrics.get("weight_countering", {})
    if wc.get("counter_balance_pct", 100) < 20:
        flags.append("  [!] Partnership — low counter-balance: limited resistance/elastic connection")
    if wc.get("partner_distance_std", 0) > 0.3:
        flags.append("  [~] Partnership — high distance variance: uneven stretch/compression dynamic")

    mu = metrics.get("musicality", {})
    if mu.get("on_beat_pct_a", 100) < 50:
        flags.append(f"  [!] Lead — on-beat articulated steps below 50% ({mu.get('on_beat_pct_a', 0):.1f}%)")
    if mu.get("on_beat_pct_b", 100) < 50:
        flags.append(f"  [!] Follow — on-beat articulated steps below 50% ({mu.get('on_beat_pct_b', 0):.1f}%)")
    if mu.get("bounce_match_a", 1.0) < 0.3:
        flags.append(f"  [~] Lead — bounce rhythm doesn't match beat (match {mu.get('bounce_match_a', 0):.2f})")
    if mu.get("bounce_match_b", 1.0) < 0.3:
        flags.append(f"  [~] Follow — bounce rhythm doesn't match beat (match {mu.get('bounce_match_b', 0):.2f})")
    if mu.get("arm_body_corr_a", 1.0) < 0.2:
        flags.append(f"  [~] Lead — low arm-body correlation ({mu.get('arm_body_corr_a', 0):.3f}): arms may not respond to body action")
    if mu.get("arm_body_corr_b", 1.0) < 0.2:
        flags.append(f"  [~] Follow — low arm-body correlation ({mu.get('arm_body_corr_b', 0):.3f}): arms may not respond to body action")

    lines += [
        SEP,
        "  SUMMARY FLAGS",
        SEP,
        "",
    ]
    if flags:
        lines += flags + [""]
    else:
        lines += ["  No significant issues flagged.", ""]

    lines += [SEP, "  END OF REPORT", SEP]


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(video_path: str, pose_data: dict, metrics: dict) -> str:
    lines = []

    _header(video_path, pose_data, metrics, lines)

    tq = metrics.get("tracking_quality", {})
    for i, label in enumerate(["lead", "follow"]):
        cov = tq.get(label, {}).get("coverage_pct", 100.0)
        if f"leg_action_{label}" in metrics:
            _leg_action_section(label, metrics[f"leg_action_{label}"], cov, lines)
        if f"body_action_{label}" in metrics:
            _body_action_section(label, metrics[f"body_action_{label}"], cov, lines)

    _weight_countering_section(metrics.get("weight_countering", {}), lines)
    _musicality_section(metrics.get("musicality", {}), lines)
    _flags_section(metrics, lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a West Coast Swing video and produce a structured report."
    )
    parser.add_argument("video",    help="Path to dance video file")
    parser.add_argument("--out",    default=None, help="Save report to this file")
    parser.add_argument("--poses",  default=None, help="Use pre-extracted poses JSON (skip extraction)")
    parser.add_argument("--skip",   type=int,   default=1,    help="Process every Nth frame (default 1)")
    parser.add_argument("--conf",   type=float, default=0.30, help="Pose detection confidence (default 0.30)")
    parser.add_argument("--save-poses", action="store_true",  help="Save extracted poses to JSON for reuse")
    args = parser.parse_args()

    video_path = args.video

    # ---- Pose extraction ----
    if args.poses:
        print(f"Loading poses from: {args.poses}")
        with open(args.poses) as fh:
            pose_data = poses_from_serialisable(json.load(fh))
    else:
        print(f"Extracting poses from: {video_path}")
        print("  (downloading yolov8n-pose.pt on first run — ~6 MB)")
        pose_data = extract_poses(video_path, conf=args.conf, frame_skip=args.skip)
        ids = pose_data.get("dancer_ids", [])
        print(f"  {len(pose_data['frames'])} frames  |  {len(ids)} dancer(s)  |  IDs: {ids}")

        if args.save_poses:
            poses_path = str(Path(video_path).with_suffix(".poses.json"))
            with open(poses_path, "w") as fh:
                json.dump(poses_to_serialisable(pose_data), fh)
            print(f"  Poses saved to: {poses_path}")

    if len(pose_data.get("dancer_ids", [])) < 1:
        print("[!] No dancers detected. Check video quality or lower --conf.")
        sys.exit(1)

    # ---- Metric computation ----
    pose_data["video_path"] = video_path   # needed for beat extraction
    print("Computing metrics...")
    metrics = compute_all_metrics(pose_data)

    # ---- Report ----
    report = build_report(video_path, pose_data, metrics)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report saved to: {args.out}")
    else:
        print()
        print(report)


if __name__ == "__main__":
    main()
