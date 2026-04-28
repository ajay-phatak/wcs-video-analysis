---
name: wcs-analyze
description: >
  Full West Coast Swing (WCS) dance video analysis pipeline. Use this skill
  whenever the user wants to analyse a dance video — whether they give a local
  file path or a YouTube URL, say "run the analysis", "how does this compare to
  the pros", "analyse my video", or anything that involves measuring leg action,
  body action, connection/countering, or musicality from a WCS video. Also
  trigger when the user asks to add a new pro reference, expand the pro
  benchmark set, or re-run a previous analysis with updated metrics.
---

# WCS Video Analysis Skill

## What this skill does

Runs the full analysis pipeline on a West Coast Swing dance video and produces:
1. A structured text report covering leg action, body action, weight/countering,
   and musicality sections
2. (Optionally) a gap-comparison table against the pro reference videos

## Project location

All Python modules live in:
```
C:\Users\wizar\OneDrive\Documents\Projects\Dance Analysis\
```
Key files:
- `pose_extraction.py` — `extract_poses(video_path)` → dict  
- `dance_metrics.py`  — `compute_all_metrics(pose_data)` → dict  
- `dance_review.py`   — `build_report(video_path, pose_data, metrics)` → str  
- `pro_reference_poses.json` / `pro_reference2_poses.json` — pre-computed pro benchmarks

The bundled runner script is at:
```
C:\Users\wizar\OneDrive\Documents\Projects\Dance Analysis\wcs-analyze-skill\scripts\analyze.py
```

---

## Step-by-step workflow

### 1. Identify the input

The user will give either:
- A **local file path** (any video format — `.mp4`, `.MOV`, etc.)
- A **YouTube URL** (e.g. `https://www.youtube.com/watch?v=…`)

If the input is ambiguous (e.g. "my latest video"), ask for the path or URL before proceeding.

### 2. Run the analysis script

```
python "C:\Users\wizar\OneDrive\Documents\Projects\Dance Analysis\wcs-analyze-skill\scripts\analyze.py" "<input>" [--compare-pros]
```

- Pass `--compare-pros` whenever the user wants to see how they stack up against
  the pros, or when it would be useful context (e.g. first-time analysis of an
  amateur video).
- The script handles downloading, pose extraction caching, metric computation,
  and report generation automatically.
- Pose extraction is slow (~3–5 min for a 3-min video at 25-60 fps). The script
  caches results as `<stem>_poses.json` next to the video so subsequent runs are
  instant. Let the user know if extraction is running.

### 3. Present the results

After the script finishes it prints the full report. Summarise the key findings:

- **What's going well** (metrics near or above pro level)
- **Top 3 gaps vs pros** (biggest deltas in the gap analysis, if run)
- **One actionable focus** — the single metric gap most worth working on right now

Keep the summary concise. The full report is already printed; don't repeat every number.

### 4. Outputs saved automatically

- `<video_stem>_report.txt` — full report, in the same directory as the video
- `<video_stem>_gap_analysis.txt` — pro comparison, if `--compare-pros` was used

---

## Adding a new pro reference video

When the user wants to add another pro video to expand the benchmark:

1. Run `analyze.py` on the new URL/file — poses are cached automatically.
2. Add the new entry to `PRO_REFS` in `scripts/analyze.py`:
   ```python
   (DANCE_DIR / "new_pro.mp4", DANCE_DIR / "new_pro_poses.json", "Pro N (BPM)"),
   ```
3. Confirm with the user that the entry is added.

---

## Key implementation details

- **JSON key normalisation**: poses files saved with `json.dumps(..., default=str)`
  store dancer dict keys as strings and numpy arrays as their string repr. The
  script's `_normalise_poses()` converts both back before passing to
  `dance_metrics`. If you ever call `compute_all_metrics` directly (not via the
  script), run `_normalise_poses` first.

- **`video_path` key**: must be set on the poses dict before calling
  `compute_all_metrics` — it's used for audio/beat extraction.

- **Poses cache naming**: `<video_stem>_poses.json` in the same directory as the
  video. Check for this file before running extraction to avoid re-running.

- **yt-dlp + ffmpeg**: uses `python -m yt_dlp` (not the CLI binary) and
  `imageio_ffmpeg.get_ffmpeg_exe()` for merging streams when yt-dlp downloads
  video and audio separately (which happens wh
