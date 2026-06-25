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

**Which dancer is the user, and what role?** The pipeline has no dance-role detection —
it just tracks two people as Dancer 1 and Dancer 2 (identities are kept stable across the
clip by an appearance/colour re-ID — see Key implementation details). To attribute the
stats you must know **which tracked dancer the user is** and **which role they dance**.
**Ask "which side do you start on, left or right?"** and pass `--me left` / `--me right`;
the script resolves the side to a Dancer ID from the first clean frame. **Also ask "do you
lead or follow?"** and pass `--role lead` / `--role follow` (default `lead`) — this is what
makes the gap analysis compare the user against the pros of their *own* role.
**Verify it landed on the right dancer** — entry-heavy clips (walk-ons, the dancers far
apart at the start) can fool the side resolver. If it's wrong, extract a clear frame,
identify the user, and re-run with **`--me-id 1`** or **`--me-id 2`** to set the Dancer
ID directly (this overrides `--me`). Default is `--me left`.

### 2. Run the analysis script

```
python wcs-analyze-skill/scripts/analyze.py "<input>" [--compare-pros] [--me left|right | --me-id 1|2] [--role lead|follow] [--partner] [--spotlight]
```

- Pass `--me left` or `--me right` to say which side the user starts on, or
  `--me-id 1|2` to name their Dancer ID directly (use when the side resolver picks the
  wrong dancer). The report header and gap analysis label that dancer **"you"**.
  Defaults to `--me left`.
- Pass `--role lead` or `--role follow` (default `lead`) to set the user's role. The
  "you" rows are then compared against each **pro of that same role** (pro leads are
  configured per clip by `lead_id` in `pro_refs.json`; the pro follow is the other dancer).
- Pass `--compare-pros` whenever the user wants to see how they stack up against
  the pros, or when it would be useful context (e.g. first-time analysis of an
  amateur video).
- Pass `--partner` to also analyse the user's **partner** — it adds a `-- PARTNER --`
  block to the gap analysis comparing the partner against each pro of the *other* role.
  Use this when the user wants their partner's analysis too. You can then bridge the
  partner's gaps to notes the same way (if the user's own lesson notes are from their
  own perspective, use universal / other-role-oriented concepts for the partner and say so).
- Pass `--spotlight` when the clip is a **full-floor spotlight/showcase** (the couple is
  meant to travel around the room). By default a clip is treated as **contained**
  (prelim/practice), where staying compact is expected — so the *Couple travel around
  room* gap row is annotated "lower expected" and should **not** be presented as a
  deficiency. With `--spotlight` that row is compared to the pro baseline normally.
  **Ask the user whether the clip is a spotlight** before running with `--compare-pros`,
  since pro references are typically spotlights and the couple-travel comparison is only
  apples-to-apples for another spotlight.
- Pass `--pose-model n|s|m|l|x` to choose the YOLOv8-pose model (default `m`). Use `l`/`x`
  for max accuracy on small or crowded figures (slower; poses are cached once). The model
  name is recorded in the poses JSON, and a mismatch with a cached file is reported.
- The script handles downloading, pose extraction caching, metric computation,
  and report generation automatically.

#### Crowded footage — seed the user's couple

The default tracker assumes the only two people on screen are the couple, so it fails on
crowded floors (prelim heats, socials) where it can't tell which two detections are the
user. If the clip has other couples/people moving, use **crowd mode** instead of `--me`:

1. **Preview:** `analyze.py "<input>" --seed-frame <seconds>` — renders a numbered preview
   `<stem>_seed.png` of everyone detected at that timestamp and exits. Pick a timestamp
   where the user and partner are both clearly visible.
2. **Show the user the preview** and ask which number is them and which is their partner.
3. **Re-extract:** `analyze.py "<input>" --seed-me-idx <n> --seed-partner-idx <n> [--compare-pros --partner]`
   — tracks only that couple out of the crowd (matches every frame against appearance +
   motion anchors built from the two seeded people; ignores other couples; leaves a dancer
   *missing* rather than swapping to a stranger when occluded). Seeding pins **dancer 1 =
   the user, dancer 2 = partner**, so `--me`/`--me-id` are not needed.

To pick the right numbers from the preview, the foreground couple is usually the largest,
most-confident, front-most boxes; background couples and mirror reflections are smaller or
off to the side. **Caveat:** seeding fixes *identity*, not *resolution* — if the couple
fills little of the frame (distant phone footage), the fine per-dancer joint/articulation
metrics stay approximate, so lead with the partnership and rhythm metrics for such clips.
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

1. Run `analyze.py` on the new URL/file — poses are cached automatically as
   `<stem>_poses.json` next to the clip.
2. **Identify which tracked dancer is the pro lead.** Don't guess from side — the
   tracker's Dancer 1 is not reliably the left dancer, and `lead_id` is **per clip**
   (the same pro can be Dancer 1 in one clip and Dancer 2 in another). Render a clear
   **mid-performance** frame, NOT the first frame: pro clips often open on a blurred
   intro/title card where the tracker boxes the audience. Pick a frame where both
   dancers' bounding boxes are large (≈70%+ of image height = the performers on the
   floor), label each tracked Dancer ID on it, and have the user confirm which ID is
   the lead.
3. Add an entry to `pro_refs.json` (see `pro_refs.example.json` for the format):
   ```json
   {"video": "pros/new_pro/clip.mp4", "poses": "pros/new_pro/clip_poses.json",
    "label": "Couple N — Event (BPM)", "couple": "Couple N", "lead_id": 2}
   ```
   Paths are relative to the project root (or absolute). `couple` groups clips: the
   gap analysis averages each couple's clips and prints a **separate section per
   couple**, so use the same `couple` string across that couple's clips. The user's
   lead stats are compared against that couple's lead; partnership metrics (posts,
   distance variance, counter-balance) are role-agnostic and unaffected.
4. **Caveat on long clips:** full-song performances (3–5 min) accumulate tracker
   identity swaps, which inflate the per-dancer timing-consistency and post-count
   numbers. Trust the body-mechanics rows from such clips; lean on shorter,
   cleanly-tracked clips for timing/post comparisons.
5. Confirm with the user that the entry is added.

---

## Reading the travel & post metrics

The report's **TRAVEL DECOMPOSITION** section breaks partnership movement into three
physically distinct kinds of travel (all normalised to body heights, BH) — don't conflate
them:

1. **Floor travel** — movement *of* the slot around the room: the couple's shared centre
   relocating across the floor. Measured from a strongly low-passed (≈1 s) centroid:
   `couple_travel_range_bh` (bounding extent, the robust headline used in the gap table) and
   `couple_travel_path_bh` (cumulative, secondary). **Spotlight-sensitive**: only compare to
   pros when `--spotlight` is set; in a contained prelim/practice clip, low floor travel is
   correct, not a gap.
2. **Slotted movement** — movement *down* the slot, per dancer: each dancer's *absolute*
   position along the slot axis, `travel_lead`/`travel_follow` → `slot_travel_range_bh` /
   `slot_travel_path_bh`. This is how far the lead and the follow each traverse the slot (the
   follow usually travels more). Measured absolutely (room frame) so lead and follow stay
   distinct.
3. **Stretch range / Compression range** — how far the centres move *after a post*; this is
   `post_max_stretch_mean` / `post_max_compression_mean`, surfaced here for grouping.

**Posts** are now detected when the connection point is still **along the slot axis** (not
fully 2-D still). Stretch and compression legitimately move the connecting hand *vertically*
(up/down) and slightly out; requiring full stillness used to cut those posts short, so the
count under-reported them. Expect noticeably more posts than the old 2-D test — that is the
intended correction. The slot axis is a PCA fit over both dancers' centres
(`travel.slot_axis_deg`); it should be near the partnership's `slot_direction_deg`.

Posts are also split into **stretch-leading** (`post_stretch_leading` — followed mainly by
stretch) vs **compression-leading** (`post_compression_leading` — followed mainly by
compression). This is *descriptive*, not a gap: it shows whether the dancer tends to post
before stretching the partner away or before compressing them in. Use it to spot a lopsided
habit (e.g. lots of stretch-leading but few compression-leading posts), not as a
higher=better score.

**Pro-baseline caveat:** floor travel is partnership-level (more trustworthy); per-dancer
slotted movement inherits the ~80% pro identity-stability caveat — hedge per-dancer pro deltas.

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
