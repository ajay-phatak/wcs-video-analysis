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
- `--pose-model n|s|m|l|x` — YOLOv8-pose model size (default `m`). Use `l`/`x` for
  max accuracy on small or crowded figures (slower; poses are cached once).
- `--output-dir <dir>` — where to save reports (default: same directory as the video).

Pose extraction is slow (a few minutes for a multi-minute clip; longer with `l`/`x`).
Results are cached as `<stem>_poses.json` next to the video; subsequent runs are instant.

---

## Crowded footage — pick your couple

By default the analyser assumes the only two people on screen are the couple. On a
crowded floor (a prelim heat, a social) that breaks down: it can't tell which two
detections are *you*. Crowd mode seeds the tracking to your couple in two steps:

```bash
# Step 1 — render a numbered preview of everyone detected at ~45s in:
python wcs-analyze-skill/scripts/analyze.py path/to/clip.mp4 --seed-frame 45

# Open <stem>_seed.png, find the number on you and on your partner, then:
# Step 2 — re-extract, tracking only your couple out of the crowd:
python wcs-analyze-skill/scripts/analyze.py path/to/clip.mp4 \
    --seed-me-idx 3 --seed-partner-idx 5 --compare-pros --partner
```

Seeding builds an appearance + motion model of the two people you point at and matches
every frame against it, ignoring other couples; when you're occluded the dancer is left
*missing* rather than swapped onto a stranger. It also pins **dancer 1 = you, dancer 2 =
partner**, so `--me`/`--me-id` aren't needed. Pick a seed frame where both of you are
clearly visible. Note: isolating your couple fixes *identity*, not *resolution* — if you
fill little of the frame, the fine per-dancer joint metrics stay approximate; the
partnership and rhythm metrics are the reliable reads from distant footage.

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

### The three-pass extraction pipeline

Pass 1 (above) handles detection, tracking, and identity. Two further passes
upgrade keypoint quality; run them in this order (each invalidates the next):

**Pass 2 — `pose_refine.py` (top-down refinement, adds feet).** Keeps pass-1's
tracking but re-estimates each dancer's keypoints with RTMPose in **Halpe-26**
format. Pass 1 sees the whole frame at ~640px, so small/distant dancers get
thumbnail-quality joints; the refinement crops each dancer from the
original-resolution frame, cutting keypoint jitter ~15–55% and adding real
foot keypoints — heels + toes at indices 20–25 (0–16 stay COCO-compatible so
all metrics work unchanged). ~40 f/s on CPU; `--mode performance` for the
larger 384×288 model; `--preview <sec>` writes a before/after overlay png.

```bash
python pose_refine.py path/to/clip.mp4 --replace-cache   # pass 1 kept as *_poses_pass1.json
```

**Pass 3 — `pose_lift.py` (3D lifting).** Lifts each dancer's 2D keypoint
sequence to root-relative 3D joints (VideoPose3D temporal model, H36M-17
format, stored per frame under `dancers3d` alongside the untouched 2D data).
2D joint angles are camera-angle-dependent — your clip and a pro clip filmed
from different positions aren't directly comparable; 3D joint angles are
rotation-invariant. Validated: 3D leg bone lengths ~3× more stable than 2D,
L/R symmetry ~1%. Arm joints are noisier than legs/torso — prefer leg/torso
angles for 3D metrics. Global travel and absolute rise/fall stay 2D (the 3D
output is root-relative, so it has no global trajectory).

```bash
python pose_lift.py path/to/clip.mp4        # augments <stem>_poses.json in place
```

**Re-running the whole library — `reextract_all.py`.** Refined metrics are NOT
comparable against pass-1 metrics (one-foot % in particular reads dramatically
higher — more accurately — with refined ankles). Whenever the pipeline
changes, re-extract your clips **and** all pro references together so
everything stays on the same measurement scale:

```bash
python reextract_all.py --dry-run    # list what would be processed
python reextract_all.py              # refine + lift every cached clip
```

Credits: pass 2 uses [RTMPose](https://github.com/open-mmlab/mmpose) via
[rtmlib](https://github.com/Tau-J/rtmlib) (Apache-2.0); pass 3 uses
[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
(CC-BY-NC-4.0, non-commercial — model definition vendored as
`videopose3d_model.py`, pretrained weights downloaded on first run).
