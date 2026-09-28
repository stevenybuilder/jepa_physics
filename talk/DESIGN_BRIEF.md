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
