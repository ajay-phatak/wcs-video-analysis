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
3. Personalised practice recommendations that **bridge each notable gap to the
   user's own Obsidian WCS lesson notes** — turning a raw stat ("you travel on
   far more steps than pros") into a pointer to what they already worked on
   ("you drilled settling with Keerigan & Mia on 2-27-26 — review those notes").
   See "Bridging gaps to your notes" below.

## Project location

All Python modules live in the project root (set via the `DANCE_ANALYSIS_DIR` env var,
or leave unset to use the current working directory):

Key files (shared code/model stays at the root):
- `pose_extraction.py` — `extract_poses(video_path)` → dict  
- `dance_metrics.py`  — `compute_all_metrics(pose_data)` → dict  
- `dance_review.py`   — `build_report(video_path, pose_data, metrics)` → str  

**Put your clips anywhere you like.** Each video keeps its own poses cache, report,
gap analysis, and practice notes alongside it. Pro references are configured via
`pro_refs.json` in the project root (see `pro_refs.example.json`); if absent, the
script auto-discovers `pros/*/` subfolders.

The runner script is at:
```
wcs-analyze-skill/scripts/analyze.py
```

---

## Step-by-step workflow

### 1. Identify the input

The user will give either:
- A **local file path** (any video format — `.mp4`, `.MOV`, etc.)
- A **YouTube URL** (e.g. `https://www.youtube.com/watch?v=…`)

If the input is ambiguous (e.g. "my latest video"), ask for the path or URL before proceeding.

**Which dancer is the user?** The pipeline has no dance-role detection — it just
tracks two people as Dancer 1 and Dancer 2 (identities are kept stable across the clip
by an appearance/colour re-ID — see Key implementation details). The user dances
**lead**, so to attribute the stats to them you must know which tracked dancer they are.
**Ask the user "which side do you start on, left or right?"** and pass `--me left` /
`--me right`; the script resolves the side to a Dancer ID from the first clean frame.
**Verify it landed on the right dancer** — entry-heavy clips (walk-ons, the dancers far
apart at the start) can fool the side resolver. If it's wrong, extract a clear frame,
identify the user, and re-run with **`--me-id 1`** or **`--me-id 2`** to set the Dancer
ID directly (this overrides `--me`). Default is `--me left`.

### 2. Run the analysis script

```
python wcs-analyze-skill/scripts/analyze.py "<input>" [--compare-pros] [--me left|right | --me-id 1|2] [--partner]
```

- Pass `--me left` or `--me right` to say which side the user (the lead) starts on, or
  `--me-id 1|2` to name their Dancer ID directly (use when the side resolver picks the
  wrong dancer). Either way the report header and gap analysis label that dancer
  **"you"** and compare it against each **pro's lead** (the pro lead is configured per
  clip by Dancer ID in `PRO_REFS`). Defaults to `--me left`.
- Pass `--compare-pros` whenever the user wants to see how they stack up against
  the pros, or when it would be useful context (e.g. first-time analysis of an
  amateur video).
- Pass `--partner` to also analyse the **follower** — it adds a `-- PARTNER --` block
  to the gap analysis comparing the partner against each pro's *follow* (not lead). Use
  this when the user wants their partner's analysis too. You can then bridge the
  partner's gaps to notes the same way (the user's own notes are the lead's, so use the
  universal/follow-oriented concepts and say so).
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

Then run **"Bridging gaps to your notes"** (below) and fold the resulting
note-based recommendations into the summary — each top gap should, where possible,
point to the user's own prior instruction for closing it.

Keep the summary concise. The full report is already printed; don't repeat every number.

### 4. Outputs saved automatically

- `<video_stem>_report.txt` — full report, in the same directory as the video
- `<video_stem>_gap_analysis.txt` — pro comparison, if `--compare-pros` was used
- `<video_stem>_practice_notes.txt` — the bridged, note-based practice
  recommendations (written by this skill via the Write tool, since the Python
  pipeline can't reach the Obsidian MCP). See below.

---

## Bridging gaps to your notes

This is what makes the analysis personal: instead of leaving the user with raw deltas,
take each notable gap and surface the user's **own prior instruction** for closing it,
from their Obsidian West Coast Swing lesson notes.

**Reference:** `references/metric_hub_map.md` — maps every stat to the relevant note
hub(s) and the concept/keyword terms to search for, and documents the category
definitions, the retrieval procedure, and the guardrails. Read it before bridging.

### When to run

After presenting a `--compare-pros` analysis (step 3). It also works without a pro
comparison — drive it off the report's `SUMMARY FLAGS` section instead of the gap table.

### Procedure

1. **Pick the gaps to bridge.** From the gap analysis, take the `▼` rows (unfavorable vs
   pros) with the largest gaps — judge by *relative* size, since metrics have different
   scales — plus anything in `SUMMARY FLAGS`. Cap at the top ~3, and always include the
   single "one actionable focus" you named. (`▲` rows are favorable — don't bridge those.)
   **Prioritise the `— you` rows and the partnership rows** (those are the user); the
   `— partner` rows are the user's partner, so only bridge those if the user asks about
   their partner. (Rows are labeled `you`/`partner` based on the `--me` side.)
2. **Map each gap** to its hub(s) + search terms using `references/metric_hub_map.md`.
3. **Find the instruction** with the Obsidian MCP tools, scoping strictly to
   `West Coast Swing/`:
   - `mcp__obsidian-mcp-connector__search_vault_simple` on the search terms — keep only
     hits whose path starts with `West Coast Swing/`. The snippet is often the bullet itself.
   - `mcp__obsidian-mcp-connector__get_backlinks` on the target hub (e.g.
     `Movement - Concepts.md`), filtered to `West Coast Swing/` sources, to confirm/expand.
   - `mcp__obsidian-mcp-connector__get_vault_file` on the best 1–3 notes to read the exact
     bullet and confirm it addresses the gap.
   - (`mcp__obsidian-mcp-connector__list_vault_files` to enumerate `West Coast Swing/` if
     needed; `…__search_vault_smart` may substitute for `_simple` if semantic search works.)
4. **Cite the lesson** by parsing instructor + date from the filename (`Keerigan 6-20-25.md`
   → "Keerigan, 6-20-25"; `keerigan and mia 2-27-26.md` → "Keerigan & Mia, 2-27-26"). For
   event/undated notes, cite the title as-is.
5. **Compose each recommendation** in this shape:
   *stat finding → the cited instruction (quoted or closely paraphrased) + which lesson &
   date → a suggested practice focus.*

### Guardrails

- **Never invent** a lesson, instructor, date, or quote. Only cite instructions that
  actually appear in a `West Coast Swing/` note.
- **Scope strictly** to `West Coast Swing/`. Never cite a `Ballroom/` note (or `Mapping
  Report.md`) — Ballroom is out of scope for now.
- If no relevant WCS instruction exists for a gap, **say so plainly**; you may add a single
  clearly-labeled *general* tip, but don't attribute it to the notes.
- If the Obsidian MCP connector is unavailable, **fall back** to the plain gap analysis and
  tell the user the notes couldn't be reached.

### Output

Fold the recommendations into the spoken summary (step 3), **and** write them to
`<video_stem>_practice_notes.txt` next to the report using the Write tool. One block per
bridged gap: the stat finding, the cited instruction(s) with lesson + date, and the
suggested practice focus.

---

## Reading the musicality metrics (song-aware)

The musicality section is **song-aware**: it characterises the music itself and then asks
whether the movement matches it. Interpret it with these principles (also encoded in
`references/metric_hub_map.md`):

- **SONG CHARACTER** (bounciness on a bouncy↔smooth axis, dynamic range, # accents detected)
  is *context, not a score*. It describes what the song asks for. **Never compare song
  character you-vs-pro** — they're dancing to different songs. Compare the **match** scores.
- **TEXTURE match** is time-resolved: a positive correlation means the dancer gets bouncier in
  punchy passages and smoother in legato ones — movement quality tracking the song.
- **MUSICAL EXPRESSION** is detected at musical accents found **throughout** the song (hits,
  breaks, stabs), not only at 8-bar phrase boundaries, and it counts expression through **any
  channel** — punctuated **feet**, a **chest**/body pop, the free **hands**, or the **head**.
  - The **dominant channel** ("via feet", "via head") is *descriptive* — it shows how the dancer
    expresses, useful for suggesting variety; it is **not** a deficiency.
  - **Partnership coverage** credits an accent that **either** partner expresses, and **framing**
    captures the lead going still to *set up the follow* to hit the moment — a legitimate musical
    choice, **not** the lead missing the accent. A real expression gap is low *coverage* (the hit
    lands for neither) or low *texture match*, not a low individual response when framing is high.

---

## Reading the articulation-quality metrics

The LEG ACTION block has an **ARTICULATION QUALITY** sub-block scoring *how well* articulated
steps (heel lifts clear of the ground) are executed — not just how many there are. It models a
generic step as **bend = preparing to move a foot → (body flight if traveling) → straighten =
weight arriving on the new foot**, and it **distinguishes the free/moving leg from the standing
leg** (they do different jobs — conflating them is misleading):

- **Free-leg prep flexion** (knee/hip degrees) — how much the *moving* leg bends *while its foot
  is free* to gather/prepare the step. This does NOT lower the body (weight is on the other leg).
- **Standing-leg knee flexion** (degrees) — how much the *weighted* leg sinks/loads; this is the
  one that actually "gets lower". (Per the per-step analysis, foot-lift, body pitch, and standing-leg
  sink are independent channels — some pros, e.g. Maria, couple foot-lift/pitch with the gather;
  others keep a steady quiet prep, so don't read a single "correct" channel into it.)
- **Free knee↔hip coordination** — does the *gathering* leg's knee and hip flex together
  (proportional chain) vs knee-only / hip-only.
- **Bend smoothness** — one clean prep→rise on the moving leg vs a jittery/segmented bend.
- **Straighten recovery** — how fully the moving leg re-straightens after the bend (the *rise* as
  weight arrives on the new foot).
- **Prep→arrival sequencing** — the bend happens while the foot is free (prep), then straightens
  after it grounds.
- **Ankle lift** — a *proxy only* (moving-foot vertical excursion / push through the ball of the
  foot). The pose model has no foot keypoint, so a true ankle joint angle can't be measured.

**Caveats:** joint angles are 2-D side-on estimates (good for relative you-vs-pro, not absolute);
**musical accents are valid exceptions** — e.g. stepping into a deep lunge to mark a hit will read
as extra-deep flexion with delayed straightening, which is intentional, not a flaw. Treat these as
aggregate tendencies over many steps, not per-step verdicts.

### Deep detail (the MOVEMENT-QUALITY & MUSICALITY DETAIL section)

Every report also includes a per-dancer **detail** section that goes beyond the medians — read
and discuss it for practice videos:

- **Bend-depth distribution** (median / p75 / p90 / max for standing- and free-leg flexion). The
  key read is **ceiling vs steady**: a wide median→p90 spread means a big dynamic range (the dancer
  *goes for it* on some steps); a low ceiling means a compressed range (rarely sinks deep). A gap
  vs pros that is small at the median but large at p90 is a *ceiling* problem, not a typical-step one.
- **Accent vs anchor**: bend depth on steps that land on a musical accent vs steps that don't. Deeper
  ON accents = the dancer loads to mark the music (e.g. Maria); deeper OFF = the depth lives on
  anchors/settles (common for leads). This is the movement×musicality intersection.
- **Amplitude regulation** (Pearson r across steps): foot-lift↔prep-flex, body-pitch↔prep-flex,
  standing-sink↔COM-drop — *how* the dancer scales step size. These are the most exploratory numbers
  (sensitive to leg occlusion and 2-D), so report them with hedging and lean on `n`/consistency.

When presenting a practice video, work these into the summary: name whether each gap is a *typical-step*
gap or a *ceiling* gap, and whether the dynamic movement is *landing on the music* or not.

---

## Adding a new pro reference video

When the user wants to add another pro video to expand the benchmark:

1. Run `analyze.py` on the new URL/file — poses are cached automatically.
2. **Identify which tracked dancer is the pro lead.** Don't guess from side — the
   tracker's Dancer 1 is not reliably the left dancer. Extract a clear frame (dancers
   close to camera, both detected) and look at who is leading, or ask the user. Then
   add the entry to `PRO_REFS` in `scripts/analyze.py` with that lead **Dancer ID**
   (`1` or `2`) as the 4th element:
   Add an entry to `pro_refs.json` (see `pro_refs.example.json` for the format):
   ```json
   {"video": "pros/new_pro/clip.mp4", "poses": "pros/new_pro/clip_poses.json",
    "label": "Pro N (BPM)", "lead_id": 2}
   ```
   Paths are relative to the project root (or absolute). The user's lead stats are
   compared against this dancer; partnership metrics (posts, distance variance,
   counter-balance) are role-agnostic and unaffected.
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
  video. Check for this file before running extraction to avoid re-running. Delete the
  cache to force re-extraction (e.g. after changing the tracker).

- **Identity tracking (appearance re-ID)**: `pose_extraction.py` keeps Dancer 1/Dancer 2
  locked to the same person across the clip using a torso **colour signature** (an
  H-S-**V** histogram — value/brightness matters: it's what separates a black top from a
  white/grey shirt). Two **frozen anchors** are built up front by clustering the torso
  signatures over all clean (full-size) frames, then every frame is assigned by spatial
  proximity + colour distance to those anchors. Freezing (vs a per-frame EMA) is
  deliberate — it stops a reflection or one bad crossing from poisoning the reference and
  swapping the identities for the rest of the clip. Without this, long clips with slot
  crossings silently swap who is "Dancer 1," blending the two dancers' per-dancer metrics.
  Verify identity is stable on new clips (e.g. compare a start vs end frame); re-ID needs
  the two dancers' torso colours to differ (they usually do).

- **yt-dlp + ffmpeg**: uses `python -m yt_dlp` (not the CLI binary) and
  `imageio_ffmpeg.get_ffmpeg_exe()` for merging streams when yt-dlp downloads
  video and audio separately (which happens when ffmpeg is not on PATH).
