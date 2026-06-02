# Metric → Notes Map

This file turns a **stat gap** from the analysis into a **search** of the user's Obsidian
West Coast Swing notes, so the skill can recommend the user's *own* prior instruction for
closing that gap (e.g. "you travel on far more steps than pros — you worked on settling with
Keerigan & Mia on 2-27-26").

It is consumed by the **"Bridging gaps to your notes"** step in `SKILL.md`.

---

## The note vault, in brief

- Lesson notes live under `West Coast Swing/`. **Only pull from this folder.** Ignore
  `Ballroom/`, `Mapping Report.md`, and everything else in the vault.
- Each lesson note is a list of bullet instructions; every bullet is tagged with one or more
  hub wikilinks: `[[Category - Layer]]`.
- There are **12 hubs** = 4 categories × 3 layers. Hubs are populated by *backlinks*, not by
  listing entries inline.
- Lesson **filenames encode instructor + date**, e.g. `Keerigan 6-20-25.md`,
  `keerigan and mia 2-27-26.md`, `3-16-25 KP.md`, `John Lindo 8-25-25.md`. Some are events
  (`Liberty 2024.md`, `ESS 2025.md`) or planning notes (`WCS focus.md`, `Questions for ...`).
  Parse the instructor/date from the filename for citation; if it's an event or undated note,
  cite the note title as-is.

### Authoritative category definitions (from the hub frontmatter — use these, not intuition)

| Category | Covers |
|---|---|
| **Movement** | leg/foot actions, footwork, body flight, swing, rise & fall, level changes |
| **Partnering** | connection: compression, tension, leverage, lead/follow communication |
| **Musicality** | timing, rhythm, syncopation, phrasing, accents, matching the music — **and pattern selection / song interpretation** |
| **Structure** | frame, posture, body tone, poise (how the body is carried/held) |

### Layers

- **Mechanics** — joint/muscle how-to (the physical execution).
- **Concepts** — reusable principles that apply across patterns.
- **Ideas** — high-level imagery / metaphors for a feeling.

> The current pose metrics observe **Movement, Partnering, and Musicality** well.
> **Structure (posture / poise / body tone) is only weakly observable** from pose data, so it
> mostly appears as a *secondary* hub on body-tone-adjacent metrics, or from a visual /
> `SUMMARY FLAGS` observation rather than a hard number.

---

## Mapping table

Each row: the stat/finding (as it appears in the gap analysis or the report's `SUMMARY FLAGS`),
the target hub(s), what the gap means in plain English, and concept/keyword terms to search for.
Hubs in *(parentheses)* are secondary — try them only if the primary hub yields nothing relevant.

| Stat / finding | Hub(s) | Gap means | Search terms |
|---|---|---|---|
| High steps/min · high weight-only or articulated **traveling** | Movement-Concepts/Ideas, Partnering-Concepts | moving more than needed; not settling | stillness, settle, anchor, "break less", economy, hold, delay |
| Low **1-foot balance %** | Movement-Mechanics/Concepts | not committing weight; rushed transfers | weight transfer, commit, balance, ball of foot, foot pressure |
| Low **rise/fall** (typical/dynamic) | Movement-Mechanics/Ideas | flat body, little level change | rise and fall, bounce, level change, compression into floor, knee bend, body flight |
| Low **knee flexion** | Movement-Mechanics | straight legs, little compression | knee bend, compression, sit, lower, leg spring |
| Low **art. free-leg prep flex** (knee/hip) | Movement-Mechanics | not bending the moving leg to gather/prepare the foot before stepping | prep the foot, gather, pick up the foot, knee bend, leg swing, collect under you |
| Low **art. standing-leg knee flex** | Movement-Mechanics | weighted leg stays straight — not sinking/loading into the floor | sit into the standing leg, compression into the floor, get low, load, drive from the floor |
| Low **art. free knee-hip coordination** | Movement-Mechanics | the gathering leg bends knee-only or hip-only, not a coordinated chain | hip hinge, knee + hip together, leg line, don't pike, stacked |
| Low **art. bend smoothness** | Movement-Mechanics/Concepts | jittery/segmented load instead of one clean bend→rise | smooth, continuous, roll through the foot, one motion, no hitch |
| Low **art. straighten recovery** / **prep→arrival sequencing** | Movement-Mechanics/Concepts | leg doesn't straighten as weight arrives, or bend mistimed vs the step | straighten the leg, push the floor away, rise, settle then send, prep the foot, weight arrival, drive from the floor |
| Low **motion smoothness** / choppy | Movement-Concepts/Ideas | choppy / staccato swing | smooth, continuous, fluid, flow, swing, connect movements |
| Block body / low **hip→shoulder fluidity** | Movement-Mechanics; *(Structure-Mechanics: body tone)* | body moves as one block, no layering | sequential, dissociation, layering, body tone, joint twist |
| Low **pitch / sway range** | Movement-Concepts/Ideas; *(Structure-Concepts: posture)* | body stays flat/upright, little shaping | pitch, lean, sway, shaping, body flight, stretch the side |
| **Posture / poise / frame** collapsed or untoned (visual or flags) | Structure-Mechanics/Concepts/Ideas | frame, posture, body tone not held | posture, poise, frame, body tone, stay tall, head position, shoulders down, carriage |
| Low **partner distance variance** / little stretch-compression | Partnering-Concepts/Mechanics | static connection, no leverage dynamic | stretch, compression, leverage, extension, slingshot, elastic, time under tension |
| Low **slotted movement range** (down the slot, per dancer) | Movement-Concepts/Ideas; *(Partnering-Concepts)* | dancer doesn't traverse the slot — staying planted instead of travelling from one end to the other | travel the slot, down the slot, send her down, post-and-travel, cover the slot, go somewhere |
| Low **floor travel** (spotlight only) | Movement-Concepts/Ideas | in a spotlight, the couple stays parked instead of moving the slot around the floor — IGNORE for contained/prelim clips (lower is expected, not a gap) | use the floor, travel the room, spotlight, cover ground, journey, stage |
| Low **stretch range / compression range** (after a post) | Partnering-Mechanics/Concepts | the post is set but little body travel comes out of it — weak elastic payoff | stretch out of the anchor, slingshot, leverage, send, expand off the post, time under tension |
| Few **posts** / low post stretch | Partnering-Concepts; *(settling also Movement-Concepts)* | no anchors to stretch from | anchor, post, settle, "directional intent", stop with intent, hold the connection |
| Low **counter-balance** | Partnering-Concepts/Mechanics | limited shared resistance | counterbalance, leverage, lean away, shared weight, resistance |
| **Connection noise** / wrong contact point | Partnering-Mechanics | elbows/shoulders absorb body action | quiet the connection, pool floaties, hands down and out, don't move the elbow, spring |
| Low **on-beat %** / poor **timing consistency** | Musicality-Mechanics/Concepts | steps off / inconsistent timing | timing, on the beat, accent, count, land on time |
| Low **bounce match** | Musicality-Concepts; *(Movement-Mechanics)* | bounce rhythm ≠ music | bounce to the beat, pulse, groove, match the texture |
| Low **music-movement tracking** | Musicality-Concepts/Ideas | energy doesn't track music | dynamics, energy, accent the music, texture, hit, build |
| Low **texture match** (move vs song) | Musicality-Concepts/Ideas; *(Movement-Mechanics)* | movement quality doesn't change with the song — bouncy in legato passages or flat in punchy ones | match the texture, bouncy vs smooth, quality of movement, what the music calls for, groove vs glide |
| Low **accent response %** / **hit intensity** | Musicality-Concepts/Ideas | not marking the musical accents/hits the song offers | hit, accent, break, stab, mark the music, punctuate, "catch the hit", big moment |
| Low **partnership accent coverage** | Musicality-Concepts; *(Partnering-Concepts)* | hits land for neither partner — nobody catches the moment | hit, accent, set up the follow, frame, "give her the moment", lead for musicality |
| Low **syncopation** | Musicality-Concepts | little rhythmic variation | syncopation, & counts, rhythm play, hold-and-go |
| Few / repetitive **6/8-count patterns** (pattern selection) | Musicality-Concepts/Ideas | thin or repetitive pattern choices for the song | pattern selection, variety, song interpretation, "what the music calls for", vocabulary |
| Few **triple steps** (footwork execution) | Movement-Mechanics/Concepts | triple-step footwork itself | triple step, footwork, anchor step |
| Stiff **arm styling** (shoulder→wrist lag, body-arm corr) | Musicality-Mechanics; *(Movement-Concepts)* | stiff arms, not body-connected/expressive | free arm, styling, arm follows body, wave, expression |

If a finding isn't in the table, route it by the **category definitions** above, then pick the
layer: Mechanics for a "how do I physically do it" gap, Concepts for a "what principle am I
missing" gap, Ideas for a "I need a feeling/image" gap.

### Interpreting the musical-expression metrics (important)

There is **no single correct channel** for expressing musicality. The analyzer detects accent
expression across **feet (punctuated steps), chest/body, free hands, and head**, and credits a hit
to whichever channel is strongest. So:

- The **dominant channel** (e.g. "via feet", "via head") is *descriptive, not a deficiency* — it
  tells you how the dancer tends to express, useful for suggesting variety, not a gap to "fix".
- **Partnership accent coverage** and **framing** capture the lead's option to go still and *set
  up the follow* to express a hit while he frames her — this counts as the moment landing, not as
  the lead missing it. A low *individual* accent response with healthy *coverage* + *framing* is a
  legitimate musical choice, not a flaw. Treat a real gap as low coverage (the hit lands for
  **neither** partner) or movement texture that doesn't track the song (low **texture match**).
- **Song character** (bounciness / dynamic range / accent count) is *context for the other rows*,
  not a score — it describes what the song asks for. Don't compare song character you-vs-pro
  (different songs); compare the **match** scores (texture match, accent response).

---

## Retrieval procedure (per gap)

1. Look the stat up in the table → target hub(s) + search terms.
2. `mcp__obsidian-mcp-connector__search_vault_simple` on 2–4 of the search terms. Keep only hits
   whose path starts with `West Coast Swing/`. The returned context snippet is often the
   instruction bullet itself.
3. To confirm/expand, call `mcp__obsidian-mcp-connector__get_backlinks` on the target hub
   (e.g. `Movement - Concepts.md`) and keep only `West Coast Swing/` sources — these are notes the
   user explicitly filed under that category.
4. `mcp__obsidian-mcp-connector__get_vault_file` on the best 1–3 candidate notes to read the exact
   bullet (and its sub-bullets), and confirm it actually addresses the gap.
5. Parse instructor + date from the filename for the citation.
6. (`mcp__obsidian-mcp-connector__search_vault_smart` may be used instead of `_simple` if semantic
   search is available; fall back to `_simple` otherwise.)

## Guardrails

- **Never invent** a lesson, instructor, date, or quote. Only cite instructions that actually
  appear in a `West Coast Swing/` note. Quote closely or paraphrase faithfully.
- **Scope strictly** to paths beginning `West Coast Swing/`. Never cite a `Ballroom/` note.
- If no relevant WCS instruction exists for a gap, **say so plainly**. You may add a single
  clearly-labeled *general* WCS tip, but mark it as general — don't attribute it to the notes.
- If the Obsidian MCP connector is unavailable, **fall back** to the plain gap analysis and tell
  the user the notes couldn't be reached.
