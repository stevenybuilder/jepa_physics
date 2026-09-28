# Design principles for the talk

Written 2026-09-28 17:20 ET. This file governs every slide in `talk/slides_v2/` and every figure in
`talk/figures_v2/`. It merges the user's brief (below, §1), the interviewer audit (`logs/DECK_AUDIT_FABLE.md`), the
palette and type system already in `talk/DESIGN_BRIEF.md`, and the presentation-design research gathered earlier
today (§7, with sources). `DESIGN_BRIEF.md` is folded into §9 below, so this is the single source for the design system and the research behind it.

## 1. What the deck is for

The deliverable: "a presentation of the findings, aimed at approximately 15 minutes. The presentation will serve as
the basis for an open discussion … Present the main methods, results, interpretations, comparisons, and limitations
clearly." The reader is an AI research lead who wrote the paper we reproduced. She reads it twice:

- **Surface read (30 seconds).** Is it well designed? Are there clear, novel experiments and headline findings, or at
  least unique thinking? Is it clean and pleasant to look at?
- **Deep read (15 minutes and the discussion).** Is she peering into a mind that thinks critically from first
  principles? Does the deck say why this problem, why this experiment, why this technique, what is new, and what it
  does not show?

The standard: the interviewer should enjoy reading it. Punchy, succinct, visually calm, and intellectually dense in
the right places only.

## 2. Three tiers of content, three levels of detail

Every result belongs to exactly one tier, and the tier fixes how much space and detail it gets.

| Tier | What | Space | Detail | Marker |
|---|---|---|---|---|
| **A. Headline findings** (3–5 in the whole deck) | Novel results the interviewer must notice: read vs use; readable everywhere but used from the object; the chord wins the endpoint, the curve wins the path; one joint edit for coupled variables | One full slide each, plus a numbered row on the "what we found" slide | Deep: the question, why we asked it, why this technique, the headline figure, the one number, the caveat that could kill it | **Numbered and bold** (1, 2, 3, 4) on the findings slide and as an eyebrow on their own slides |
| **B. Supporting results** | Reproductions, controls, negatives, the three README variables' bake-off, novel variables that did not become headlines (time step, relational motion, conceptor, geodesic, twin) | One line each on a summary slide, or one shared slide | Minimum viable: the claim and one number; the rest in notes and REPORT | Plain |
| **C. Methods and setup** | Model, data, probes, splits, arms, controls | One design-rationale slide and one parameters slide | Minimum viable on the slide (the design logic and the three leaks); every parameter in the notes | Plain |

Rule: never give a Tier B or C item Tier A space, and never let a Tier A item shrink to a table cell.

## 3. The four "why"s every headline slide must answer

In the title, the one takeaway line, and the notes, a headline slide answers:

1. **Why this problem** matters for a world model (planning, control, trust), not just for probing.
2. **Why this experiment** is the one that answers it (what alternative design would have failed, which leak it closes).
3. **Why this technique** (spline, token swap, predictor readout, joint sheet) rather than the obvious one.
4. **What it means** for how the model represents physics, in plain words, and what would change our mind.

The paper's own open questions (C.1.4 binding; "the predictor's use of these codes" never tested; manifold steering
untested at held-out physical values) are named where our finding answers them.

## 4. Structure that follows the deliverable

Methods → results → interpretations → comparisons → limitations, in that order, with a motivation slide first and
a discussion close. About 22 main slides plus appendix slides after the close, 15 minutes: 30–45 s for Tier B and C
slides, 75–90 s for Tier A slides.

1. Cover: the thesis in one line.
2. Why this question (motivation; readability is not use; the two papers' foundation and their open questions).
3. Methods: the design rationale (evidence ladder and the three leaks), then the parameters.
4. Part 1 divider; the three Part 1 result slides (what training adds; what the zone is; what the probe count measures).
5. Part 2 divider; results in the order shape → edit → judge → generalise; the "beyond the three variables" slide.
6. Interpretation: what the papers established and left open; what we asked and found (numbered headlines).
7. Comparison: the arms in plain words; against the two papers.
8. Limitations, and what would change our mind. Citations. Close on the opening question with two questions for discussion.
9. Appendix: the full numeric table and any dense reference material, after the close.

Read the titles alone: they must tell the story without the slides.

## 5. Slide rules

- **One claim per slide, as the title.** An assertion of 8–14 words in one grammar throughout. The title states the
  physical claim, never the instrument ("A straight edit between two headings passes through no heading at all", not
  "The spline stays on the ring in the ring plane").
- **Glance test.** The point is clear in 3 seconds. Signal to noise about 9:1.
- **≤ 30 visible words** excluding title and axis labels; ≤ 5 words per label line; nothing under 24 px; body 28–32 px.
- **One figure or one big number, large.** Never several small panels. Never bullets; use a table, a card row, a big
  number, or one sentence.
- **Numbers earn their place.** A number appears only if it changes what the reader concludes; the rest live in the
  notes and `NUMBERS.md`, each with a file and key path.
- **No jargon without its meaning.** On slides: "a straight edit", "an edit along the curve", "the paper's probe
  method", "a gate". Arm names (raw chord, smoothed chord, probe-QR, conceptor, COAST) live in notes.
- **Physics, not instrument.** Say what the measured quantity means about how the model represents motion and what
  it implies for steering a world model.
- **No LLM slop.** No "It's not X, it's Y"; no faux-insight titles; no triads for their own sake; no colon drama; no
  hedged filler; no walls of text; no emoji; no cards with coloured fills or left-border accents.
- **Honesty markers.** A result that is still running or unverified appears in notes only, marked "running". Negatives
  are stated as findings, not apologies. Every caveat that could kill a headline appears on its slide.
- **Notes carry the script.** Full spoken sentences, ~150 words per minute, sources first.

## 6. Visual system (summary; the full system is §9)

- Palette with fixed meanings: navy `#14213D` ink and dark slides; off-white `#FBFBF8` pages; terracotta `#C8553D`
  for the trained model, the curved edit, and the point of the slide; grey `#9AA1AB` for untrained, nulls, unedited;
  steel blue `#3D6A9E` for straight edits and probe methods; ochre `#C9A227` for the paper's reference values; soft
  text `#5B6472`. No gradients. Grey for context, one bright colour for the point.
- Type: Source Serif 4 600 for titles and big numbers; IBM Plex Sans for body; sizes 160 / 96 / 56 / 32 / 24 only.
- Grid: `padding:128px 128px 160px`; title at the top margin on every content slide; body y ≈ 300–920; one 24 px
  footer line at `bottom:64px`. Repeated elements keep their place; slides of one kind share markup.
- Whitespace on purpose. Do not fill space with logos, cards, or footer prose. Fill about 70% of the column or anchor
  the body with `space-between`.
- Figures (figures4papers style, Tufte data-ink): minimal spines, no gridlines, direct labels instead of legends, one
  accent colour, axis labels in plain words, consistent font, no title inside the image (the slide title is the
  claim), 1600×900 at 2× DPI, made from the JSON by script, never by hand.
- Contrast ≥ 4.5:1 for text (3:1 above 44 px); colours that must be told apart differ in lightness, not hue alone.

## 7. Research this rests on

- Alley, assertion–evidence slides (Penn State): sentence-assertion title, visual evidence, no bullets.
- Duarte, the glance test and "5 secrets to displaying data": 3-second clarity; grey context, one colour for the point.
- Figma presentation guides: the 5/5/5 rule, whitespace, two-tone palettes, hierarchy by size.
- Garr Reynolds, Presentation Zen design tips: restraint, empty space, few colours used consistently, one image large.
- Tufte, data-ink ratio and chartjunk: remove gridlines, boxes, in-plot titles, legends where lines can be labelled.
- Gamma visual-hierarchy guide: size carries importance; the largest element is read first; fixed type scale.
- Fatahalian, "Tips for giving clear talks": why before what; section slides as re-entry points; explain every figure's
  axes then the one thing to see; close on the bigger picture; no more than 3 sections in a 15-minute talk.
- Slides craft guidance of the artifact type (one idea per slide; titles introduce, they do not deliver a verdict;
  128 px margins; nothing under 24 px; budget vertical space; parallel design; notes carry the script).
- ChenLiu-1996/figures4papers: clean, consistent, minimal scientific figures.
- Scientific-talk convention: methods only to the depth needed to trust the result; results as claims with the number
  that carries them; limitations as what would change the conclusion, not a list of everything imperfect.

## 8. Checklist before publishing

- [ ] Title sequence alone tells the story (motivation → open questions → how we test → Part 1 → Part 2 → what is new → limits → close).
- [ ] 3–5 headline findings are numbered and bold, each with a full slide answering the four whys.
- [ ] The three README variables each have a stated finding; every novel variable (motion sheet, time step, two-disk
      relational motion, binding by depth) is listed and labelled with its finding and whether it is new.
- [ ] No slide over 30 words; no text under 24 px; no arm jargon without meaning; no instrument-only titles.
- [ ] Every visible number has a `NUMBERS.md` row; running results are in notes only.
- [ ] Figures remade in the palette and figures4papers style; axis labels in plain words.
- [ ] Negatives and caveats present on the slides they belong to; limits slide says what would change our mind.
- [ ] Close returns to the opening question and ends with questions for discussion.

## 9. The design system and its sources (formerly `talk/DESIGN_BRIEF.md`, merged verbatim)

# Talk design brief (Phase 1)

Goal: a 15-minute deck that an author of the reproduced paper enjoys watching. One finding per slide, stated as a sentence, shown with one large figure or one big number. The script lives in speaker notes.

## (a) Rules and their sources

1. **One assertion per slide, written as the title.** Use a full sentence of 8–14 words on at most two lines. Titles read in order should summarise the talk. Sources: [Alley, PSU AE instruction set](https://cpb-us-e1.wpmucdn.com/sites.psu.edu/dist/7/13153/files/2008/10/Assertion-Evidence-Slides-Instruction_Set.pdf); [Fatahalian, Tips for Clear Talks, tips 8–9](https://graphics.stanford.edu/~kayvonf/misc/cleartalktips.pdf).
2. **The evidence is visual: a figure, a diagram or a short table. Never bullets.** Allow at most 1–2 call-outs per figure. Source: [Alley AE instruction set](https://cpb-us-e1.wpmucdn.com/sites.psu.edu/dist/7/13153/files/2008/10/Assertion-Evidence-Slides-Instruction_Set.pdf); AE slides measurably improve comprehension ([Alley research page](https://writing.engr.psu.edu/research.html)).
3. **Three-second glance test.** If a slide's point is not clear within 3 s, it is noise. Aim for 9:1 signal to noise. Sources: [Duarte Glance Test](https://www.duarte.com/resources/guides-tools/the-glance-test/); [HBR/Duarte](https://hbr.org/2012/10/do-your-slides-pass-the-glance-test).
4. **Keep ≤ 30 words on a slide, not counting the title and axis labels.** Put no more than 5 words on any label line. Sources: [Figma, 5/5/5 rule](https://www.figma.com/resource-library/presentation-ideas/); [Gamma: keep text short, slide is not the script](https://gamma.app/insights/beyond-bullet-points-smarter-slide-design-in-gamma).
5. **Make each figure big: one image large, not several small.** Remake paper figures for the talk and cut any panel that does not serve the title. Sources: [Reynolds tip 6](https://www.garrreynolds.com/design-tips); [Fatahalian tip 8, corollary](https://graphics.stanford.edu/~kayvonf/misc/cleartalktips.pdf).
6. **Grey for context, one bright colour for the point.** Put a contrasting colour on the one bar or line that carries the message. Source: [Duarte, 5 secrets to displaying data](https://www.duarte.com/blog/display-data-in-presentations/).
7. **Restrain, reduce, emphasise.** Remove gridlines, boxes, in-plot titles and legends where the lines can be labelled directly. Sources: [Tufte data-ink/chartjunk (Stasko notes)](https://faculty.cc.gatech.edu/~stasko/7450/16/Notes/tufte.pdf); [Reynolds tip 9](https://www.garrreynolds.com/design-tips).
8. **Use empty space on purpose.** Do not fill space with logos, cards or footer prose. Sources: [Reynolds tip 1](https://www.garrreynolds.com/design-tips); [Figma "don't fear the white space"](https://www.figma.com/resource-library/presentation-ideas/).
9. **Use two typefaces and a fixed size hierarchy.** Size carries importance, and the largest element is read first. Sources: [Gamma visual hierarchy guide](https://gamma.app/explore/content/guides/how-gamma-builds-clean-modern-presentations-with-visual-hierarchy); [Figma typography](https://www.figma.com/resource-library/typography-in-design/).
10. **Use a few colours, and use them the same way throughout.** Sources: [Reynolds tip 12](https://www.garrreynolds.com/design-tips); [Figma, two-tone palettes](https://www.figma.com/resource-library/presentation-ideas/).
11. **Give the why before the what, and use section slides as re-entry points.** A 10–15 min talk should have no more than 3 sections. Sources: [Fatahalian tips 6 and 10](https://graphics.stanford.edu/~kayvonf/misc/cleartalktips.pdf); [Alley AE instruction set, step 2](https://cpb-us-e1.wpmucdn.com/sites.psu.edu/dist/7/13153/files/2008/10/Assertion-Evidence-Slides-Instruction_Set.pdf).
12. **Explain every figure aloud: axes, then the one thing to see.** This goes in the notes, not on the slide. Source: [Fatahalian tip 7](https://graphics.stanford.edu/~kayvonf/misc/cleartalktips.pdf).
13. **Close on the bigger picture, not a list of problems.** Source: [Fatahalian tip 11](https://graphics.stanford.edu/~kayvonf/misc/cleartalktips.pdf).
14. **Format house rules.** One idea per slide. Use a 4–5 size type scale with nothing under 24 px. Keep 128 px margins and the heading at the same height on every content slide. Title style must not be verdict-drama or "It's not X, it's Y". Put the script in `<aside>`, never on the slide. Source: `artifact-type/reference/craft.md`, `layout.md`.

## (b) Design system

**Palette.** Each colour means one thing, on every slide and in every figure.

| Role | Hex | Meaning |
|---|---|---|
| Ink / dark slides | `#14213D` navy | text; cover, dividers, closing background |
| Page | `#FBFBF8` off-white | content-slide background |
| Card | `#FFFFFF`, border `1px solid #E4E2DC`, radius 12 | figure cards only |
| **V-JEPA 2 / spline (the focal result)** | `#C8553D` terracotta | trained-model curves (Part 1) and the spline arm (Part 2); big numbers that are the finding |
| **Random init / nulls / unedited** | `#9AA1AB` grey | random-init ViT-L, random bases, random curves, unedited baseline |
| **Paper's value** | `#C9A227` ochre (bands, markers, dashed lines only; its label text stays navy) | the paper's reported depth or count, shown as a reference |
| **Straight chord / linear edit** | `#3D6A9E` steel blue | chord arm and probe-QR arm (Part 2) |
| Soft text | `#5B6472` | captions, source line |

- No gradients. On a navy slide the dark-mode text is `#FBFBF8` and the accent is `#E07A5F` (terracotta, lightened for contrast).
- No accent-background statement slide, because terracotta carries data meaning. Big numbers do that job instead.

**Type.** Headings and big numbers use Source Serif 4 at weight 600. Body and labels use IBM Plex Sans at weight 400 or 500. The deck uses five sizes:

| px | Use |
|---|---|
| 160 | big number, line-height 1 |
| 96 | cover and divider title |
| 56 | assertion title (≤ 2 lines, line-height 1.1, letter-spacing −0.5px, width ≤ 1500) |
| 32 | body and big-number caption |
| 24 | eyebrow (caps, letter-spacing 3px, terracotta), axis callouts, source line |

**Grid.** Padding is `128px 128px 160px`. The title is pinned by flow at y = 128 on every content slide, with a 48 px gap under it. The body occupies y ≈ 300–920. The source line is one 24 px line, pinned at `left:128; bottom:64`, and gives the result-file stem only. There is no footer prose and no slide number text beyond "n / 18".

**Archetypes.**
1. **Cover (navy).** A 96 px title, a 32 px subtitle and the name/date line.
2. **Divider (navy).** A 24 px eyebrow "PART 1", a 96 px topic, and one 32 px line saying what the section asks.
3. **Assertion + figure.** The title, then a figure card at full width (1664 × ≤ 600) or 1120 wide plus a 480 px rail. The rail holds one big number, a 32 px caption and at most one call-out.
4. **Big number.** Two or three numbers at 160 px, each with a 32 px caption, placed on one baseline. Used for before/after pairs such as 37/64/88 → 1/1/1.
5. **Comparison table.** At most 4 rows × 3 columns at 32 px. The winning cell's text is terracotta or steel blue per the palette. There is no zebra striping.
6. **Diagram.** Inline SVG artwork (≤ 52 KB, no `<text>`), with labels as `<p>` over it.
7. **Closing (navy).** One sentence at 56 px, three short lines at 32 px, then "Questions".

**Figures.**
- One panel per slide, shown ≥ 1100 px wide on a white card with 32 px padding.
- Re-render every hero figure from the `results/*.json` files with a new `talk/make_talk_figs.py`. Leave `scripts/make_figures.py` untouched.
- Render conventions:
  - Deck palette, and IBM Plex Sans if it is available to matplotlib (fall back to DejaVu Sans).
  - Rendered at 2× the display box.
  - Tick labels ≥ 24 px at display size.
  - No in-plot titles or suptitles, no boxed legends: label lines directly.
  - Spines only on the left and bottom, and no grid, or at most a faint y-grid.
- Axis meaning goes in the notes. On-figure text is limited to two call-outs, written as `<p>` over the card where possible.

**Numbers.** A headline number appears once, as a big number (160 px) with a ≤ 8-word caption. Supporting numbers go in the notes. Degrees keep the ° sign; ranges use an en dash.

## (c) Slide plan (18 slides, ≈ 50 s each; dividers ≈ 10 s)

Number sources: **S** = current `talk/slides/<name>.html` and **R1** = REPORT.md §1.

| # | Title (assertion, ≤ 12 words) | Single visual | On-slide text (≤ 30 words) | Big numbers [source] | Notes carry |
|---|---|---|---|---|---|
| 1 | *Cover:* Direction, speed and acceleration inside a frozen V-JEPA 2 | none (navy) | "Reproducing Joseph et al., then steering along the direction ring" · name · Sept 2026 | – | framing, the two papers |
| 2 | One orange disk moves; the encoder stays frozen | 4-frame strip cropped from `stimuli_paper_layout_examples.png` (one row) | ViT-L/16, 256², 26 readout points · 1,500 / 1,536 / 1,536 clips · ridge probes, 80/20 split | – [S setup] | held-out clips / values / context; 70/30 rerun changes no verdict [R1] |
| 3 | *Divider:* Part 1 · Reproducing the probes | – | "Where is each variable, how many directions, can we steer it?" | – | – |
| 4 | Pooled probes decode all three variables from the first block | re-rendered `fig1` left panel + random-init direction (grey) | V-JEPA 2 vs random-init ViT-L, direction | **0.875** V-JEPA 2 block 1 · **0.848** random init [S p1-layers] | speed 0.983, accel 0.977; peaks; paper's 0.2–0.6 pre-zone phrased neutrally [S] |
| 5 | Per-patch probes separate the trained network from a random one | re-rendered per-position mean R² vs layer, V-JEPA 2 (terracotta) vs random (grey), supplied clips | Probe each patch position, then average | **0.96** by block 6 · **≤ 0.39** random init [R1] | random pools to 0.86–0.88; VideoMAE matches pooled [S p1-why] |
| 6 | On a harder render, the per-patch onset lands at the paper's depth | `fig1j` left panel re-rendered; paper's transition as ochre band | Textured floor, shading, smaller disk; three render seeds | **9 / 8 / 8** onset per seed [R1] | largest rise at points 4–6; inflection 5.7–5.9; half-frame 0.7 at point 1 [R1] |
| 7 | Erasing each variable takes tens of probe directions | `fig2_inlp` top-left panel re-rendered (direction), random-removal line grey | Direction, speed, acceleration at the paper's layer | **37 / 39 / 41** probes · random removal leaves R² **0.980** [S p1-inlp] | paper-protocol K 46/45/47; the speed-vs-direction result depends on the stop rule [S, R1] |
| 8 | Whitened, direction is a rank-2 code stretched over many dimensions | big-number archetype | Points 9 / 12 / 22 | **37 / 64 / 88** → **1 / 1 / 1** [S p1-dims] | after rank-2 erasure ridge ≈ 0, MLP 0.37–0.82; planted controls [S] |
| 9 | Five learned probes steer direction to within 9° | `fig3_steering` left panel, learned (terracotta) vs random rank-2N basis (grey); paper's ≈ 20 as ochre tick | Held-out evaluation probe; steered clips from test | **8.7°** at N = 5 · random rank-2N **57.2°** [S p1-steer] | the paper's Adam basis needs 18 against the paper's ≈ 20 [R1]; nulls |
| 10 | One probe steers once the edit follows the activation covariance | big-number pair | Euclidean edit vs covariance-weighted edit, same probe | **78°** → **3.2°** · random plane median **4.2°** [R1] | specificity: 0.03 vs 4.2 m/s off-target [S p2-compare] |
| 11 | *Divider:* Part 2 · Steering along the ring | – | "What changes when the variable lives on a curve?" | – | Goodfire recipe [S p2-ring] |
| 12 | Direction centroids form a ring; speed and acceleration form lines | re-rendered `fig4g` point-12 panel (centroids coloured by angle, fitted ellipse) | Class means in the ring plane, point 12 | **−0.98** circular correlation · axis ratio **0.68–0.89** [S p2-ring] | label-free angle works at point 12 only; chord midpoint radius 0.09 [S] |
| 13 | Steering targets are held-out directions the spline never saw | SVG diagram: ring, held-out 45° arc shaded, spline path (terracotta) and chord (steel blue) | 45° arc · 8 directions · 15 arcs · 632 knot / 480 probe / 300 steered clips | – [S p2-heldout] | why held-out values matter (sagitta); frozen verdict rule; controls [S] |
| 14 | On held-out arcs, the spline path stays on the ring | `fig4_waypoint_readout…L12_contiguous` left panel, spline vs chord only | Readout radius along the path | **16 of 16** arcs · min radius **+0.26** [S p2-arc] | point 22 +0.28; the hollow exists in the ring plane only, not in 64-D [S, R1] |
| 15 | At held-out endpoints, the raw-centroid chord lands closer | comparison table | Endpoint error, headline arc | chord **4.7°** vs spline **9.7°** (pt 12); **3.6°** vs **10.7°** (pt 22) [R1] | 16-arc means +1.3° ± 1.8 / +2.3° ± 2.0; ties with the smoothed-knot chord [R1] |
| 16 | Only late edits change the predictor's forecast | HTML bar chart, forecast error to target at point 22: unedited 92.1 / random 85.7 grey · smoothing spline 44.2 · chord 27.2 · probe-QR 17.1 · interp. spline 11.3 | Edits at points 2, 8, 12 are undone within four blocks | **11.3°** vs **92.1°** unedited [S beyond] | predictor-native readout 8.9°; along-path spline−chord −13°, −31° at ≥ 135° [S, R1]; encoder-output caveat |
| 17 | These results hold for one disk on one floor | two columns of 3 short lines: *Scope* / *Next* | Scope: one stimulus family, one 45° arc for the forecast, one probe readout. Next: more renders, training checkpoints, multi-object scenes | – [S limits] | full limitations list; compute $0.19 / 38 min; 227 tests [S repro] |
| 18 | *Closing (navy):* The next question is what the model does with each variable | 3 lines, 32 px | Detection is easy on this stimulus · Geometry decides how to edit · Depth decides whether the edit is used | – | three defended claims [R1]; "Questions" |

Open item for Phase 2: slide 16 could have a second panel showing the forecast along the path. Use it only if a per-waypoint forecast file exists in `artifacts/`/`results/session2_*`; otherwise −13° stays in the notes. Every number in the plan above appears verbatim in S or R1. No new numbers.

## (d) Tone rules

- The author is in the room. Slides never say "her", "the paper contradicts itself", "inconsistencies", "fails" about the paper, or "I was right". Write "the paper" or "Joseph et al.", and only in the notes.
- A difference is stated as a finding with its condition, e.g. "Under ridge on the same clips, speed needs fewer probes". It is never framed as a correction.
- The paper's values appear as an ochre reference mark beside ours, without commentary.
- Negative results get the same assertion format as positive ones (slide 15). No hedging adjectives, no "surprisingly", no "crucially".
- No verdict titles ("X wins", "reproduced in part"), no "It's not X, it's Y", no rhetorical questions in titles except dividers.
