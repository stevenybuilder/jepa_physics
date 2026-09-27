# Priorities: what Sonia expects at minimum, and what makes it stand out

Two tiers. Tier 1 is not negotiable: every item ships, with a figure and a results file, before any Tier 2 item is
shown. Tier 2 is ranked; the cut line moves with time. Status as of 27 Sep 2026, 19:00 ET: Part 1 core and extras done at the final code (leak fixed, N grid, paper layer 9, Adam sequences with centred targets); Part 2 core done at all layers/designs with verdicts; remaining: GPU session 2 (propagation, predictor, time-reversed, hard stimuli, VideoMAE), then the report refresh.

Sources: README (what is asked), the physics paper's own conventions (what she takes for granted), the Goodfire
paper's A.3–A.7 and §5, the fidelity check against both PDFs, `lit_review.md` §1 and its "table stakes" item,
`nonobvious_components.md`, `PART2_RATIONALE.md`.

## What the reviewer says she values (her essay and the World Mechanics post, read 27 Sep)

Sources: soniajoseph.ai, "World models and interpretability are two sides of the same coin"; her LinkedIn post for
World Mechanics ("founding research team … interpretability, training dynamics, and physical AI"; "the physical world
gives us ground truth that language often does not"; "inherently interpretable and controllable foundation models").

- **Detection is not the bar; use is.** "Traditional interpretability might tell us those representations exist. Yet
  the pilot can still crash the plane if those representations fail to combine correctly." Locating a variable with a
  linear probe is the starting point, not the result. → Every Part 1 probe result is paired with a control that asks
  whether the variable is *used* (steering, off-target, propagation), and the talk leads with rungs 3–5, not rung 1.
- **The test she names is ours to run.** "If we can steer that velocity representation and observe corresponding
  counterfactual changes in future predictions, then the model starts looking less like a statistical predictor and
  more like a simulator." → Tier 2 S2 (predictor readout against rendered counterfactual twins) is the single item
  most aligned with what she wrote. It goes in the GPU session regardless of the cut line.
- **Her open question is exactly what the random-init control answers.** "Do foundation models actually develop
  cognitive maps / Internal World Models, or can they succeed through more stimulus-response strategies?" → The
  finding that a random ViT-L linearly decodes direction, speed and acceleration from block 1 on this data, and that
  training buys precision and a compact ~46-dim code rather than availability, is a direct, honest answer for this
  stimulus class. Frame it as her question, not as a failed reproduction.
- **She is comfortable with negatives against her own strong claims.** "Some of our findings in JEPA point against
  the strongest versions of this hypothesis." "The jury is still out." → Report the non-reproduction of the emergence
  zone and the missing sawtooth plainly, with the sample-size and data-regime analysis, and do not soften them.
- **Ground truth from physics, and controllability.** The variables here have exact ground truth (metadata), and the
  ask is control. → Held-out-value steering, both directions, off-target readouts, and the predictor test are the
  controllability evidence; a probe R² is not.
- **Training dynamics** is in the job description. → We have only two points on that axis (random init, final
  checkpoint), but the contrast is the training-dynamics story available without intermediate checkpoints: state it
  that way, and name the intermediate-checkpoint version as the follow-up.
- **Tone.** Enthusiastic about mechanism, allergic to overclaiming, expects the question "what does the model *do*
  with it" after every "the model *has* it".

## Tier 1: the minimum a reviewer who wrote the paper expects

### Part 1 (reproduction of methodology and qualitative findings)

| # | Item | Why it is expected | Status |
|---|---|---|---|
| 1.1 | Layer-wise linear probes on mean-pooled tokens at every layer for direction (sin/cos), speed, acceleration, plus the Cartesian pairs; x-axis layer fraction; fold mean ± SD | Fig. 2, App. B; the README's first bullet | DONE (results/p1a_*, fig1) |
| 1.2 | A stated rule for "where each variable becomes available", with onset, peak and decline reported per variable (90 % of max, bootstrap CI on the onset layer) | README says "identify where each variable becomes available"; the paper's three-part finding | done |
| 1.3 | Direction treated as circular everywhere: sin/cos targets, atan2 decode, circular MAE, a (ŝ, ĉ) scatter coloured by θ at three layers, radius reported | Paper §3.2, C.7; README Part 2 hint | done |
| 1.4 | Selectivity controls on the layer curves: shuffled labels, random-init ViT-L, pixel baseline | Any reviewer; in a fixed-camera scene raw pixels can match V-JEPA 2 (lit_review §1.8) | DONE (random-init, pixels, trajectory, random-feature floor, shuffled, disk-pool, paper-scale) |
| 1.5 | Iterative nullspace probing exactly as C.11: QR, project out, refit, the paper's stopping thresholds, 2K vs K, at the emergence layer and per layer (dimension vs depth) | App. C.11, Figs. 4c, 22, 23 | DONE at onset/8/9/peak, nested + paper protocols, leak fixed (p1b_*, fig2/fig2b) |
| 1.6 | INLP read for dimensionality **and** redundancy: sawtooth vs smooth decay (direction vs speed), the paper's within-15° accuracy per round, and a random-removal band behind the curve | README asks for "dimensionality and redundancy"; Fig. 23; the band is what makes "tens" a claim | DONE (within-15° per round, random band under the same protocol; no sawtooth under ridge, jagged 2× K under the literal Adam recipe for both variables) |
| 1.7 | Multi-probe subspace steering exactly as C.12: V from the probes, least-squares c\*, held-out evaluation probe, MAE-to-target and MAE-to-true vs N, single probe fails, ~20 reach target | App. C.12, Fig. 24; README's third bullet | DONE at onset/9/peak on the N grid (p1c_*, fig3/fig3b/fig3c incl. _paper); strict-eval extra on the box |
| 1.8 | Held-out cuts both ways: subspace from train, evaluation probe on clips it never saw, steered clips never in the subspace | README: "held-out data not used to construct the subspace" | DONE (wm.provenance block in every results JSON) |
| 1.9 | The acceleration confound (all clips start at rest; the paper's too) stated on the slide, not fixed silently; direction reported per motion type | Honesty; App. A.1.2 | in spec; per-motion-type panel done |
| 1.10 | One deviations table (ridge vs Adam with a parity check, split, input size, hidden-state points, 64 directions) | She will ask | done (spec §7b; recipe check dba3dc4: ridge ≥ Adam on all three variables) |
| 1.11 | Reproducibility: committed split file, seeds, frame hashes, versions, TF32 off, every slide number from a results JSON | Her own repo standards | done |

### Part 2 (Goodfire method applied honestly)

| # | Item | Why it is expected | Status |
|---|---|---|---|
| 2.1 | Activation manifold per A.3: PCA-64 on train, one centroid per value, natural cubic spline for speed/acceleration, **periodic** spline for direction, intrinsic angle from atan2(PC2, PC1) unsupervised and checked against θ | A.3; README: "circular structure of direction" | done |
| 2.2 | The ring shown: centroids in PC1–PC2 coloured by θ, closed loop or not, per layer | The first thing she will look for | DONE (fig4_centroid_plane_*; unsupervised angle recovered at 12/22, flagged at 8) |
| 2.3 | Manifold vs linear steering at matched endpoints with K waypoints, orthogonal complement preserved, both arms editing the same subspace (matched support) | A.6; the fidelity check found the linear arm edited the complement, being fixed | DONE (p2_steer_*, matched support, all arms) |
| 2.4 | The linear path through the ring's centre shown for large shifts; readout angle and radius at every waypoint, not the endpoint | Goodfire Fig. 4; endpoints coincide by construction | DONE (waypoint angle+radius, K=50; fig4_waypoint_readout_*) |
| 2.5 | Off-manifold energy and isometry with a stated M_y, using Goodfire's own no-output recipe (§5 Eq. 9, B.1: softmax over unsquared L2 distances to 128 spline points in PCA-64, τ = 0.5; circular at the steered layer, stated on the figure), geodesic vs geodesic | A.5, A.7, §5 | DONE (Eq. 9 with floor, entropy, τ sweep, caveat on figures) |
| 2.6 | Held-out label values never used as knots or in the subspace; evaluation probe on disjoint clips; results vs angular shift | README: "meaningful held-out steering evaluation" | DONE (scattered, contiguous, extrapolation; held-out context) |
| 2.7 | A random-curve / shuffled-centroid control with ≥ 20 draws and the spline's empirical rank | Without it spline-vs-line is a demo | DONE (≥20 draws, endpoint-matched, reflected/projected, rank) |
| 2.8 | Cheap geometry pre-checks before any spline (knot-subsampling curvature, spread vs bend, participation ratio, cone check, BF16 repeat) and a planted-ring positive control | Lets a null be a null | DONE incl. planted ring (p2_planted_ring_direction_L12) |
| 2.9 | Negatives kept and reported; nothing from the skip list | Her paper reports its own negatives (speed has no sawtooth) | policy |

## Tier 2: what would make it stand out, ranked (impact × feasibility, from lit_review.md)

| Rank | Item | Precedent / gap | Cost | Status |
|---|---|---|---|---|
| S1 | **Held-out circular steering done properly**: contiguous 45° arc held out, error vs \|Δθ\| in 15° bins, 0°→180° via 90° vs via 270° (only a manifold expresses both), readout radius collapsing along the chord | No manifold-steering paper (Goodfire, her group's 2609.01551, GAGA) evaluates on held-out values | pooled, minutes | DONE (contiguous arc, all designs, verdicts) |
| S2 | **The predictor as behavioural readout**: render counterfactual twin clips, encode context frames only, edit, run the predictor, score recovery R on predicted future tokens; propagation heatmap steer-layer × read-layer alongside | Her essay asks for exactly this; nobody has judged an edit to any JEPA by its predictor | GPU session 2, < 1 h | spec §6.5; code after layer choice |
| S3 | **"How many dimensions is direction?" as four estimands** with a planted ring: literal K, whitened K, post-LEACE-2 decodability, harmonic spectrum of the 64 centroids | Jin et al. 2608.10566 criticise her count by name; her blog names a "harmonic basis" | pooled, minutes | in progress |
| S4 | **Fewer-probes bake-off at matched edit norm**: probe-QR vs centroid transport vs rank-2 ring rotation vs spline vs snap, linear and MLP evaluators on disjoint clips | Her blog's open question | pooled, minutes | DONE (bake-off at matched norm, linear + MLP evaluators) |
| S5 | **Ring, cone or velocity plane**: radius vs speed, Procrustes to (cos θ, sin θ) vs (v cos θ, v sin θ), speed readout along a chord, Cartesian-vs-polar onset (her Table 1 claim, never plotted) | Paper asserts polar dominates without the comparison | pooled | DONE (velocity plane: ring, radius saturates in speed; Cartesian onset first) |
| S6 | **Direction-vs-speed subspace angles** (C.4 method) and the off-target readout | Her Table 3 never measures this pair | pooled | in progress |
| S7 | **Controls Goodfire lacks**: matched support (its linear arm erases the residual), reflected and projected curvature arms, matched delivered dose, BF16 repeat, shared-vs-specific Δ | jepa_steering's registered protocol; Oozeer 2605.24942 | pooled | in progress |
| S8 | **Time-reversed clips** as a direction-specific control (same occupancy, opposite direction) | Lit review table stakes; cheap and sharp | GPU session 2 | spec §6.2 |
| S9 | Straightening and the oscillation question from 2609.01551 (curvature relative to pixels on `timepool`; PCA random-walk null) | Her group's stated open question | pooled | not started; cut first |

## Order of execution once activations land

1. Tier 1 Part 1 in order 1.1 → 1.7 with 1.4 controls; figures 1–3. Decide the layer.
2. Tier 1 Part 2 pre-checks (2.8), the ring figure (2.2), then 2.1–2.7 at that layer.
3. S1, S3, S5, S6, S7 on stored activations (they share code with Tier 1 and add minutes).
4. GPU session 2 in one batch: S2 (propagation + predictor), S8 (time-reversed extraction), item 6 attentive probe only if triggered.
5. S4, then S9 if time remains.

## Cut line

If time runs short, cut from the bottom of Tier 2 upward. Never cut a Tier 1 item; if one cannot be finished, say so
on the slide rather than omit it.
