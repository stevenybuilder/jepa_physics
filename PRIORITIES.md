# Priorities: what Sonia expects at minimum, and what makes it stand out

Two tiers. Tier 1 is not negotiable: every item ships, with a figure and a results file, before any Tier 2 item is
shown. Tier 2 is ranked; the cut line moves with time. Status as of 27 Sep 2026, 16:30 ET: activations extracted and verified; step 1 and its controls, step 2 (direction) and
the Part 2 pre-checks have run on real data (outcomes in spec.md §7); step 3 running; engineers landing fixes from the
fidelity and QA passes.

Sources: README (what is asked), the physics paper's own conventions (what she takes for granted), the Goodfire
paper's A.3–A.7 and §5, the fidelity check against both PDFs, `lit_review.md` §1 and its "table stakes" item,
`nonobvious_components.md`, `PART2_RATIONALE.md`.

## Tier 1: the minimum a reviewer who wrote the paper expects

### Part 1 (reproduction of methodology and qualitative findings)

| # | Item | Why it is expected | Status |
|---|---|---|---|
| 1.1 | Layer-wise linear probes on mean-pooled tokens at every layer for direction (sin/cos), speed, acceleration, plus the Cartesian pairs; x-axis layer fraction; fold mean ± SD | Fig. 2, App. B; the README's first bullet | run on real data (results/p1a_*) |
| 1.2 | A stated rule for "where each variable becomes available", with onset, peak and decline reported per variable (90 % of max, bootstrap CI on the onset layer) | README says "identify where each variable becomes available"; the paper's three-part finding | done |
| 1.3 | Direction treated as circular everywhere: sin/cos targets, atan2 decode, circular MAE, a (ŝ, ĉ) scatter coloured by θ at three layers, radius reported | Paper §3.2, C.7; README Part 2 hint | done (radius: in progress) |
| 1.4 | Selectivity controls on the layer curves: shuffled labels, random-init ViT-L, pixel baseline | Any reviewer; in a fixed-camera scene raw pixels can match V-JEPA 2 (lit_review §1.8) | random-init and pixel/trajectory run (results/p1a_*_random, _pixels, _trajectory); shuffled + disk-pool running |
| 1.5 | Iterative nullspace probing exactly as C.11: QR, project out, refit, the paper's stopping thresholds, 2K vs K, at the emergence layer and per layer (dimension vs depth) | App. C.11, Figs. 4c, 22, 23 | direction run at onset (2) and peak (22); speed/accel running; random-model flag being added |
| 1.6 | INLP read for dimensionality **and** redundancy: sawtooth vs smooth decay (direction vs speed), the paper's within-15° accuracy per round, and a random-removal band behind the curve | README asks for "dimensionality and redundancy"; Fig. 23; the band is what makes "tens" a claim | done; within-15° metric in progress |
| 1.7 | Multi-probe subspace steering exactly as C.12: V from the probes, least-squares c\*, held-out evaluation probe, MAE-to-target and MAE-to-true vs N, single probe fails, ~20 reach target | App. C.12, Fig. 24; README's third bullet | running (results/p1c_*) |
| 1.8 | Held-out cuts both ways: subspace from train, evaluation probe on clips it never saw, steered clips never in the subspace | README: "held-out data not used to construct the subspace" | done; provenance fields (split hash, seed, commit) being added to result JSONs |
| 1.9 | The acceleration confound (all clips start at rest; the paper's too) stated on the slide, not fixed silently; direction reported per motion type | Honesty; App. A.1.2 | in spec; per-motion-type panel done |
| 1.10 | One deviations table (ridge vs Adam with a parity check, split, input size, hidden-state points, 64 directions) | She will ask | spec §7b; Adam check in progress |
| 1.11 | Reproducibility: committed split file, seeds, frame hashes, versions, TF32 off, every slide number from a results JSON | Her own repo standards | done |

### Part 2 (Goodfire method applied honestly)

| # | Item | Why it is expected | Status |
|---|---|---|---|
| 2.1 | Activation manifold per A.3: PCA-64 on train, one centroid per value, natural cubic spline for speed/acceleration, **periodic** spline for direction, intrinsic angle from atan2(PC2, PC1) unsupervised and checked against θ | A.3; README: "circular structure of direction" | done |
| 2.2 | The ring shown: centroids in PC1–PC2 coloured by θ, closed loop or not, per layer | The first thing she will look for | pre-checks run at 8/12/22 (ring found by the chart; unsupervised angle check being fixed for mirrored rings) |
| 2.3 | Manifold vs linear steering at matched endpoints with K waypoints, orthogonal complement preserved, both arms editing the same subspace (matched support) | A.6; the fidelity check found the linear arm edited the complement, being fixed | in progress |
| 2.4 | The linear path through the ring's centre shown for large shifts; readout angle and radius at every waypoint, not the endpoint | Goodfire Fig. 4; endpoints coincide by construction | in progress |
| 2.5 | Off-manifold energy and isometry with a stated M_y, using Goodfire's own no-output recipe (§5 Eq. 9, B.1: softmax over centroid distances, τ = 0.5), geodesic vs geodesic | A.5, A.7, §5 | in progress |
| 2.6 | Held-out label values never used as knots or in the subspace; evaluation probe on disjoint clips; results vs angular shift | README: "meaningful held-out steering evaluation" | scattered done; contiguous arc in progress |
| 2.7 | A random-curve / shuffled-centroid control with ≥ 20 draws and the spline's empirical rank | Without it spline-vs-line is a demo | in progress |
| 2.8 | Cheap geometry pre-checks before any spline (knot-subsampling curvature, spread vs bend, participation ratio, cone check, BF16 repeat) and a planted-ring positive control | Lets a null be a null | run at 8/12/22 and speed 12; planted ring pending |
| 2.9 | Negatives kept and reported; nothing from the skip list | Her paper reports its own negatives (speed has no sawtooth) | policy |

## Tier 2: what would make it stand out, ranked (impact × feasibility, from lit_review.md)

| Rank | Item | Precedent / gap | Cost | Status |
|---|---|---|---|---|
| S1 | **Held-out circular steering done properly**: contiguous 45° arc held out, error vs \|Δθ\| in 15° bins, 0°→180° via 90° vs via 270° (only a manifold expresses both), readout radius collapsing along the chord | No manifold-steering paper (Goodfire, her group's 2609.01551, GAGA) evaluates on held-out values | pooled, minutes | in progress |
| S2 | **The predictor as behavioural readout**: render counterfactual twin clips, encode context frames only, edit, run the predictor, score recovery R on predicted future tokens; propagation heatmap steer-layer × read-layer alongside | Her essay asks for exactly this; nobody has judged an edit to any JEPA by its predictor | GPU session 2, < 1 h | spec §6.5; code after layer choice |
| S3 | **"How many dimensions is direction?" as four estimands** with a planted ring: literal K, whitened K, post-LEACE-2 decodability, harmonic spectrum of the 64 centroids | Jin et al. 2608.10566 criticise her count by name; her blog names a "harmonic basis" | pooled, minutes | in progress |
| S4 | **Fewer-probes bake-off at matched edit norm**: probe-QR vs centroid transport vs rank-2 ring rotation vs spline vs snap, linear and MLP evaluators on disjoint clips | Her blog's open question | pooled, minutes | in progress |
| S5 | **Ring, cone or velocity plane**: radius vs speed, Procrustes to (cos θ, sin θ) vs (v cos θ, v sin θ), speed readout along a chord, Cartesian-vs-polar onset (her Table 1 claim, never plotted) | Paper asserts polar dominates without the comparison | pooled | in progress |
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
