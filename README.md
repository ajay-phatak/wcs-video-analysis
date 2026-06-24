# WCS Video Analysis

A self-contained Python pipeline for analysing West Coast Swing dance videos.
Drop in your own practice clips and pro references — no configuration needed beyond
pointing it at your files.

## What it measures

The pipeline extracts multi-person poses with YOLOv8 (COCO-17 keypoints; no foot
keypoint) and computes four metric categories:

**1. Leg action**
Rise & fall (typical and dynamic), one-foot balance percentage, step counts broken
down by traveling-articulated vs weight-only, and a full **articulation-quality**
sub-block (see below).

**2. Body action**
Centre-of-mass motion and motion smoothness.

**3. Weight / countering**
Partner distance variance, post count, and counter-balance events.

**4. Musicality (song-aware)**
The pipeline analyses the audio to characterise the song itself (bounciness,
dynamic range, accent locations throughout the track) and then asks whether the
movement matches it:
- **Texture match** — time-resolved correlation: does the dancer get bouncier in
  punchy passages and smoother in legato ones?
- **Bounce match** — beat-rhythm alignment.
- **On-beat articulated steps %**, timing consistency, syncopation %.
- **Accent response** — expression detected at musical accents found throughout the
  song via any channel (feet, chest/body, free hands, head). Partnership coverage
  credits an accent that *either* partner expresses.

### Articulation-quality detail

The LEG ACTION block includes an ARTICULATION QUALITY sub-block that distinguishes
the **free/moving leg** from the **standing/weighted leg** (they do different jobs):

- **Free-leg prep flexion** (knee/hip degrees) — how much the moving leg bends
  while its foot is free to gather the step.
- **Standing-leg knee flexion** — how much the weighted leg sinks/loads; this is
  what actually lowers the body.
- **Free knee↔hip coordination**, bend smoothness, straighten recovery, and
  prep→arrival sequencing.

Every report also includes a per-dancer **MOVEMENT-QUALITY & MUSICALITY DETAIL**
section with bend-depth distributions (median/p75/p90/max), accent-vs-anchor
comparisons, and amplitude-regulation correlations.

---

## Setup

```bash
pip install -r requirements.txt
```

YOLOv8 pose weights (`yolov8n-pose.pt`) are downloaded automatically on first run.

---

## Adding your own pro references

Create `pro_refs.json` in the project root (copy `pro_refs.example.json` as a
starting point):

```json
[
  {
    "video":   "pros/couple_a/clip_one.mp4",
    "poses":   "pros/couple_a/clip_one_poses.json",
    "label":   "Couple A — Clip One (86 BPM)",
    "couple":  "Couple A",
    "lead_id": 2
  }
]
```

`lead_id` is which tracked Dancer ID (1 or 2) is the pro **lead** in that clip.
Dancer IDs are re-assigned on every extraction **and are per-clip** — the same pro
can be Dancer 1 in one clip and Dancer 2 in another — so identify the lead from a
clear **mid-performance** frame (the first frames are often an intro/title card where
the tracker boxes the audience) and verify after each re-extraction.

`couple` is a grouping key. The gap analysis groups clips by `couple`, averages each
couple's clips together, and prints a **separate gap section per couple** instead of
one pooled average — so you can benchmark yourself against specific couples you want
to emulate, and add multiple clips of a couple to firm up their reference. Use the
same `couple` string across that couple's clips.

If `pro_refs.json` is absent, the script auto-discovers `pros/*/` subfolders — each
subfolder is treated as one couple and **all** its clips with a matching
`*_poses.json` are used, with `lead_id` defaulting to 1 and a printed reminder to verify.

---

## Running an analysis

```bash
# Basic analysis of a local video
python wcs-analyze-skill/scripts/analyze.py path/to/your_video.mp4 --me left

# With pro comparison
python wcs-analyze-skill/scripts/analyze.py path/to/your_video.mp4 --compare-pros --me left

# YouTube URL (downloads automatically)
python wcs-analyze-skill/scripts/analyze.py "https://www.youtube.com/watch?v=..." --compare-pros

# Also analyse the follower's metrics
python wcs-analyze-skill/scripts/analyze.py path/to/your_video.mp4 --compare-pros --partner
```

**Key flags:**
- `--me left|right` — which side you start on; resolved to a Dancer ID from the
  first clean frame. Default: `left`.
- `--role lead|follow` — the role you dance (default `lead`). Your stats are
  compared against each pro of the **same** role, so a follower analysing their own
  video is benchmarked against the pro follows, not the leads.
- `--me-id 1|2` — override auto-detection and name your Dancer ID directly (use
  when entry-heavy clips fool the side resolver).
- `--compare-pros` — print gap analysis vs your pro references.
- `--partner` — also show your partner's comparison (vs each pro of the other role).
- `--spotlight` — mark a full-floor showcase clip (otherwise treated as contained).
- `--output-dir <dir>` — where to save reports (default: same directory as the video).

Pose extraction is slow (~3–5 min for a 3-min video). Results are cached as
`<stem>_poses.json` next to the video; subsequent runs are instant.

---

## Output files

All outputs are saved next to the input video:

| File | Contents |
|---|---|
| `<stem>_report.txt` | Full structured report |
| `<stem>_gap_analysis.txt` | Pro comparison table (with `--compare-pros`) |
| `<stem>_practice_notes.txt` | Note-bridged practice recommendations (written by the Claude skill) |

---

## Obsidian notes bridging (optional)

When used via the bundled Claude skill (`wcs-analyze-skill/`), the pipeline can
bridge each gap to your own prior WCS lesson notes in Obsidian via the Obsidian MCP
connector. This is entirely optional — the pipeline works standalone without it.

---

## Pose model

YOLOv8-pose, COCO-17 keypoints. There is **no foot keypoint** in COCO-17; ankle
lift is estimated as a proxy from the moving-foot's vertical excursion. Joint angles
are 2-D side-on estimates — good for relative you-vs-pro comparisons, not absolute
biomechanical measurements.
