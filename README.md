# WCS Video Analysis

A pipeline for analyzing West Coast Swing dance videos and comparing your dancing to pro references. Built with YOLOv8 pose estimation, OpenCV, and a custom set of WCS-specific metrics covering leg action, body action, weight transfer / countering, and musicality.

**Status:** In progress — vibe-coded, personal-use grade. Reports generate cleanly; gap comparisons are descriptive but not yet calibrated against any formal scoring rubric.

## What it does

1. Extracts per-frame multi-person pose data from a video (YOLOv8-pose).
2. Computes WCS-specific dance metrics from the pose timeseries.
3. Generates a structured text report and (optionally) a gap-comparison table against one or more pro reference videos.

## Setup

```bash
pip install -r requirements.txt
```

You also need ffmpeg on PATH for audio extraction:

- Windows: `winget install Gyan.FFmpeg`
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg` (or distro equivalent)

The pose model (`yolov8n-pose.pt`) is committed to the repo so first-run is offline.

## Add your own videos

This repo intentionally ships without any video data. Drop your own files in the project root:

- **Your dance video** (the one being analyzed) — anywhere, just pass its path to the analyzer.
- **Pro reference videos** — any number, named `pro_reference*.mp4`. The comparison logic globs against this pattern. Pulling from YouTube with `yt-dlp` works fine.

## Run an analysis

```bash
python pose_extraction.py path/to/your_video.mp4   # writes <name>.poses.json
python dance_review.py path/to/your_video.mp4      # writes <name>_report.txt
```

Or use the included Claude Code skill at `wcs-analyze-skill/`.

## File map

- `pose_extraction.py` — `extract_poses(video_path)` returns a pose-timeseries dict.
- `dance_metrics.py` — `compute_all_metrics(pose_data)` returns the metric dict.
- `dance_review.py` — orchestrates the full pipeline and writes the report.
- `wcs-analyze-skill/` — Claude Code skill packaging.
- `yolov8n-pose.pt` — pre-trained pose-detection model.
