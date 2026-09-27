# Lessons from jepa_steering for the V-JEPA physics take-home

Rewritten 27 Sep 2026. Two sources. (1) The **public** repo, github.com/stevenybuilder/jepa_steering at commit 8151517
(14 Sep). It is the basis of the first version of this file, and its paths are relative to the repo root. (2) The
user's **local** folder, `/Users/stevenyang/Documents/jepa_steering`, cited as `js/`. It holds the root notes, the
fresh-v2 analysis outputs and the gitignored `docs/archive/`, none of which reached the public repo. A plain `grep`
skips `docs/archive/`; use `command grep`. The *why* of Part 2 is in `PART2_RATIONALE.md`; this file does not repeat it.

**Verdicts.** APPLIES = use as written. ADAPT = the idea transfers, the details change. DOES NOT APPLY = this project
has no counterpart. The difference behind most ADAPT and DOES NOT APPLY verdicts: jepa_steering edited an
*action-conditioned predictor* whose outputs fed a CEM planner, so its final endpoint was task success over 96
scenarios. Here we probe a *frozen video encoder* that has no actions, no planner and no output distribution; the
endpoints are probe readouts and later-layer activations over ~1,500 clips per dataset. Two consequences follow.
Statistical power is cheap here, where there it was the binding limit. And "behaviour" has to be defined; there it
came for free.

## 1. Pitfalls, most severe first (public repo; verdicts added)

| # | What went wrong there | How it would show up here | Rule (spec §) | Verdict |
|---|---|---|---|---|
| 1 | Push-T split by rollout ID put all 185 families in fit, dev and holdout (`PUSHT_LINEAGE_CORRECTION.md`) | clips sharing a start position or seed in train and test | group ids; `validate_split` raises (§3) | ADAPT: the clips are independent renders, so the leakage risk is label values and dev reuse, not families |
| 2 | 160 of 12,600 MetaWorld rows were byte-identical (`METAWORLD_LINEAGE_CORRECTION.md`) | duplicate clips | sha256 decoded frames (§3) | APPLIES: done, 0 duplicates in all three sets (`results/qa_*.json`) |
| 3 | FP32 cubic beat linear at every block; BF16 reversed it (`CONTROLLED_GEOMETRY.md`) | INLP K, QR bases, spline-vs-line residuals | fp32, TF32 off, float32 `meanpool` (§2) | APPLIES, and most directly to Part 2's curvature test |
| 4 | Wrapper silently differed from upstream (`POINTMAZE_TRAINING_SAMPLER_CORRECTION.md`) | no-resize preprocessing hides a bug | assert intended deviation (§2, §8) | APPLIES |
| 5 | 5 of 286 goal images had wrong hashes (`METAWORLD_STIMULUS_REPAIR.md`) | activations out of step with videos | frame hash in each file (§2) | APPLIES, cheaply |
| 6 | One frozen random direction per task beat the learned edit on Reach (`MATCHED_RANDOM_CONTROL_AUDIT_20260911.md`) | "steering works" with no random comparison | ≥ 20 random draws, two kinds of null (see §7) | APPLIES; the local folder sharpens it (§7) |
| 7 | Dose matched on average, not per window: 49% of cubic windows passed (`PATHWAY_GEOMETRY.md`) | mean norm drift fine, single clips blow up | per-clip ‖Δ‖ (§5.3) | APPLIES |
| 8 | "1,152 rows, not 1,152 contexts" (`CONTROLLED_GEOMETRY.md`) | CIs over clips when the unit is the label value | bootstrap by label value | ADAPT: this applies to *geometry* claims (64 centroids); per-clip readouts can use clips |
| 9 | "Best observed edit" labelled post hoc | CV-chosen layer or K reported as a result | label selections; test read once (§3) | APPLIES |
| 10 | A published 48.2 was never treated as an effect (`DROID_BASELINE_RECONCILIATION_AUDIT.md`) | our R² vs the paper's | paper numbers are qualitative only | APPLIES |

## 2. Practices, helpers, and spec corrections already made (public repo)

Practices, all APPLIES unless marked: hash-order split (`src/offline_study/core/protocol.py::study_split`); manifest
guard (`validate_manifest`); tensor digests (`data/inventory.py`); fail fast on degenerate cases
(`fitting/operator_fit.py::_unit`); a results checker (`scripts/check_public_results.py`, ours `check_results.py`);
zero-dose identity tests; protocol JSON committed before analysis (ours `splits/split_v1.json`); undefined stays
undefined (disk-pool NaN); separate endpoints (decodable → steerable → propagated). Helpers:
`analysis/mechanism/common.py` (`paired_bootstrap`, `holm`), `representation_geometry.py` (`random_subspace`,
`principal_angles_deg`, `participation_ratio`). Spec corrections already applied: float32 `meanpool`; fp32 with TF32
off; an "intended deviation" test in place of processor parity; frame hashing; a matched random steering control.

## 3. Scientific findings from the public repo, re-checked against the local folder

| Finding (public repo) | Verdict | Local-folder correction or nuance |
|---|---|---|
| Curvature is real in FP32 and vanishes in BF16 | APPLIES | Dose normalisation *also* reverses it: raw cubic fidelity error was "99.89% lower", but at the requested dose it was "16.03% higher" (per `js/docs/PATHWAY_GEOMETRY.md`). Compare splines at matched delivered dose |
| Curvature didn't matter for behaviour (≤ 0.026% of MSE) | ADAPT | There is no behaviour to fail here. The analogue is "curvature doesn't change later-layer agreement" |
| The four-anchor test only sees second order | APPLIES | The README calls the centre point "a held-out action" (`js/README.md` L24). It is interpolation by construction |
| "Rank 4 and 8 met the rule; rank 1 did not" | **CORRECTED** | On Reach, rank 1 passed the effect and matched-random gates (1.899%) and failed only equivalence to the best arm (`js/rank 4 intervention.md`). And the delivered edit's participation ratio (1.37–3.07) roughly matches the random control's (e.g. 3.01 vs 3.07 on PointMaze), "consistent with the dimensionality being set by the calibration procedure rather than by the learned directions" (`js/analysis/out/fresh-v2/representation_geometry/representation_geometry.md` L140). A rank count can be a property of the fitting procedure |
| The edit is ~99% one constant shift | APPLIES | The applied mean direction had cosine 0.91–0.96 with the intercept column, i.e. mostly a constant push (same file) |
| "No partial spatial support passed" | **CORRECTED** | Those sweeps used **rank 1**, so they cannot identify the rank-4 support (`js/Rank Edit.md`, "How the existing findings constrain the choice") |
| "Learned was never better than random" | **CORRECTED** | True for behaviour. False offline: the fixed-response learned edit beat its matched random by 1.203% [0.962, 1.445] and 1.475% [1.185, 1.765] of native forecast error (`js/paper/extended_abstract.md` L57). Random edits supplied "a third to a half of the native-relative gain" (`js/docs/MECHANISM_HYPOTHESES.md` L118). The honest line: *specific at the readout, not at the outcome*. This also contradicts `PART2_RATIONALE.md`'s "none of it changed forecasts more than a matched random edit did" |
| Better forecasts didn't change decisions | ADAPT | The local fresh-v2 analysis adds that forecast–success sign concordance was 5/19 cells, ρ = −0.04 (`js/analysis/out/fresh-v2/forecast_decision_outcome/`). Here, read it as: probe gains at layer L need not propagate |
| Earlier blocks corrected more (B0 3.03% → B5 0.475%) | ADAPT | This was "intervention susceptibility, not a unique physics layer" (`js/docs/MECHANISMS.md`). Here, sweep the steering layer across the emergence zone and do not pick one |
| Second-read divergence 42/100, 59/100 | APPLIES | This is the reason for the later-layer readout (nonobvious §5.3) |
| Attention maps, search entropy, factorials, operator search | APPLIES (don't) | The archive adds a stop list; see §9 |

## 4. From the local folder: reasoning, dead ends, and what we would do differently

**The trajectory the public repo hides.** jepa_steering began as a *hazard × action mechanism search* on RoboCasa
egg scenes. It pivoted to driving (protocol v0.6–v0.9) and ended on JEPA-WM MetaWorld steering. The first two were
abandoned as structured nulls:

- *Egg loop.* The model reproduced the action effect (recovered fraction 0.63) but not hazard specificity. The best
  patch recovered 0.1%, a bilinear probe scored AUROC 0.500, and the Jacobian "sonar" scored 0.16, *below* its random
  control at 0.32. V-JEPA 2-AC reproduced 0% of the interaction. Cause as stated: the egg was "100% beyond the DROID
  p95", i.e. out of distribution (`js/docs/archive/legacy-branches-2026-09-04/root-plans/LOOP1_SUMMARY.md`).
- *Driving and the "Sonar" plans.* The post-mortem memo is the clearest what-we'd-do-differently document: "seven
  protocol versions, about eighteen instruments … each added after a null"; "Hypothesis-first conjunction target … A
  null cannot say which conjunct failed"; "Instrument after every null; statistics outrunning signal" (every site at
  the permutation floor p = 1/2001); "Stereotyped stimulus → rank-1 template"
  (`js/docs/archive/legacy-branches-2026-09-04/handoffs/xie_lens_2026-09-03.md` L19–22).
  `WORLD_MODEL_SONAR_PLAN.md`, a ten-module "Frankenstein Sonar", was archived unrun after its first cell finished at
  1/15 successes and failed class support (`js/docs/archive/deferred-threads-2026-09-05/snapshot/`).
- *Main line.* The development gain did not hold on fresh data. Reach refined edit minus native was +6.25 pp in
  development (`js/wm-approaches.md` §6) and 0.00 pp on 96 fresh scenarios, with all 48 contrasts including zero
  (`js/fresh_confirmation_results.md`). The panel could resolve about 25 pp against a pre-registered useful effect of
  5 pp: "a real effect at the useful-effect scale would usually be missed by this design"
  (`js/analysis/out/fresh-v2/regime_report/regime_report.md`). An illustrative power calculation needed about 1,466
  paired scenarios (`js/matched random control.md` §7).
- *An earlier warning, ignored.* A full-spatial correction cut forecast MSE by 26.83% (visual) and 49.35% (proprio),
  yet moved two action choices in physically worse directions (`js/Rank Edit.md`, "What the findings support so far").

**Written down, never run** (`js/docs/MECHANISM_HYPOTHESES.md` §B): refit the operator to the planner's own goal
distance (H4); per-iteration CEM scores (H3); read/write validation of one direction (H7); switch the edit off after
the first divergence (H5). Also unrun: an orientation-only random control (`js/matched random control.md` §4), an
action-conditioned coefficient predictor, and an outcome-trained basis (`js/Rank Edit.md`, Designs 2–3).

| What they concluded | Verdict here | What we do |
|---|---|---|
| "The decisive comparison is selectivity, not success rate … Monotone dose over {0.25, 0.5, 1.0}" (`xie_lens_2026-09-03.md` L68) | APPLIES | Every steer reports target change, off-target change, and a dose curve |
| Stop adding an instrument after a null; "the next null gets a stimulus or a scale change" (same memo, L93) | APPLIES | Fixed analysis list in `spec.md`; a null ends that thread |
| "Stereotyped stimulus → rank-1 template"; copy-delta baseline beat the model 0.92 vs 0.68–0.83 (`cross model design jepa.md` F1, F4) | ADAPT | A single disk on black is stereotyped. The pixel baseline and random-init ViT are the template controls |
| "COAST had clean outcome labels before fitting geometry; we tried to answer several harder questions at once" (`root-plans/vla_learnings.md` L91) | APPLIES | Our labels are native and exact. Fit geometry only after layer curves exist |
| Design too small to resolve the effect it was registered to detect (`regime_report.md`) | ADAPT | Power is not our bottleneck (1,500 clips, 64 values), but centroid-level claims have n = 64 or 16 |
| The edit basis contains its own readout: "principal angles … 0.0°, 0.0°, 0.0° … the edit writes along the axes it reads" (`representation_geometry.md` L60) | APPLIES | The same risk as scoring with the subspace's own probe. The evaluation probe must be fit on disjoint clips, and its alignment with V reported |
| Win frequency ≠ average: 93/128 families positive yet a negative mean (`js/findings.md`); Reach rescued 23, broke 17 (`js/wm-approaches.md` §6) | APPLIES | Report per-clip improved and worsened counts beside mean MAE |

## 5. Circular structure and held-out evaluation: what jepa_steering already knew

**Cyclic variables appeared twice, both in archived threads; neither became a steering result.** The main line,
Push-T, Wall and PointMaze, used open scalars only. The DROID "orientation" action error is a wrap-unaware sum of
absolute deltas and was never primary (`js/src/offline_study/tasks/droid/droid_contract.py` L40–44).

1. **Egg loop (2 Sep).** A circular chart for hazard bearing was "at chance (only two bearings per scene)", so the
   test could not have succeeded (`js/docs/archive/legacy-branches-2026-09-04/handoffs/HANDOFF.md` L438).
2. **Reach-Wall geometry map (5 Sep).** A PEZ-style screen with "circular direction targets" decoded the hand's
   realised XZ motion direction on 150 discovery trajectories. It was never steered, and the thread was deferred
   the same day (details in §6, P2).

This corrects the working assumption that jepa_steering "never had a cyclic variable". Still, the closed-spline,
intrinsic-angle and chord-through-centre reasoning comes from the physics paper and Goodfire, not from prior work.
What does transfer is the machinery and one result: a circular chart, a rotating-code detector, and a **planted
rotating-code positive control that passed** (`LOOP1_SUMMARY.md` L24, L36). Copy that control before trusting any
"no ring" or "no curve" result.

| What they knew | Source | Verdict | Consequence here |
|---|---|---|---|
| "Held-out" means three things: excluded from fitting; excluded from development evaluation; untouched by development. "Calling all three simply 'held-out' obscures the evidence" | `js/methods_comparison.md`, "Which inputs were unseen" | APPLIES | Every held-out claim names its sense (`spec_updates_proposed.md` #2) |
| Reshuffling does not undo exposure; new seeds are not new source families | `js/AGENTS.md`; `js/push T recommendation.md` | APPLIES | Part 2 choices are made on train folds; test is read once for Part 2 |
| "Response audit" families were held out *only from calibration*, "not an independent test" | `js/docs/FIXED_RESPONSE_RANK4.md` | APPLIES | A probe's own CV fold is not a held-out steering test |
| The held-out point was the symmetric centre, which is pure interpolation; earlier curvature targets "had already appeared elsewhere in exploration" | `js/docs/CURVED_ACTION_RESPONSE_FOLLOWUP.md` | APPLIES | Every-4th-value holdout is interpolation over 11.25° gaps. Add contiguous-arc and extrapolation designs |
| "No unseen-action family"; circular test failed on sparse coverage | `LOOP1_SUMMARY.md`; `HANDOFF.md` L438 | APPLIES | 64 angles is dense. Held-out *contexts* exist too: the speed set covers 0.25–4 m/s, while the direction set covers 1–7 |
| Leakage found: row-level Push-T split, byte duplicates, a dropped seed, bf16 leaking ~1e-3 of a removed component | `js/docs/EXPERIMENT_PLAN.md`; `handoffs/caft_inspired_report_2026-09-03.md` L37 | APPLIES | INLP projections in fp32; check the residual probe after projection |
| Interchange: an H3-only swap recovers R = 0.50; persistent replacement gives R = 1; "one-block replacement can create inconsistency with the other five blocks" | `js/docs/ACTION_COUNTERFACTUAL.md`; `js/docs/INTERVENTION_MECHANISM_AUDIT.md` | ADAPT | Edit all 8 time steps vs some; compare against a real clip at θ\* at every later layer |
| Cyclic donor interchange (i → i+1 mod 300) beat random by only 0.007–0.0084 Spearman | `js/docs/ACTION_CONDITION_SPECIFICITY.md` | ADAPT | The Part 2 transplant: swap the ring coordinate between clips and read later layers |

## 6. Nonlinearity: the full record

The public repo reduces this to "curvature is real in FP32, vanishes in BF16, and has no behavioural effect". The
local record holds six investigations, P1–P6. Tags: **(a)** measured curvature, **(b)** nonlinear probes, **(c)**
nonlinear edits and their effect, **(d)** precision, **(e)** rotating, circular or periodic structure. "Stands" means
nothing later contradicted it; "qualified" means a later test narrowed it. `arch/` = `js/docs/archive/legacy-branches-2026-09-04/`.

**P1. Egg loop, JEPA-WM DROID predictor on RoboCasa, 2 Sep** (`arch/root-plans/LOOP1_SUMMARY.md` L24–29, L36;
`arch/handoffs/HANDOFF.md` L436–438; the method docstring survives only in
`js/scripts/cgs_pilot/__pycache__/geometry_localize.cpython-39.pyc`).
- (e) *Rotating-code detector.* The linear localizer projects each scene onto the **global** mean direction. That is
  "blind to a relational code whose direction rotates … if the interaction direction is f(bearing) with f covering
  the circle, the global mean is ~0." The fix compared projection onto the mean of the k covariate-nearest scenes
  (local) against the global mean, added RSA against bearing, and fitted a circular model. Result: 0/324 entries
  local > global. At the one candidate site, global t = 3.9 vs local t = 2.8 (local − global −0.26, CI excludes 0),
  so the code is "not rotating". RSA against bearing was significant at 12/324, which is chance. The planted
  rotating-code control passed.
- (b) kNN, RBF kernel ridge and Jacobian-local readouts beat linear at 6/324 entries (chance ≈ 16). The circular
  chart (`CircularFactorized`: X ≈ μ + A[cos θ, sin θ], phase decoded as atan2 on the fitted plane, harmonic readout;
  from the `geometry_models` bytecode) was at chance with two bearings per scene. A bilinear probe scored AUROC
  0.500. Sliced-W2 found "nothing beyond the first two moments". TwoNN intrinsic dimension was 4–5.
- (c) Edits stayed on-manifold (Mahalanobis +≤ 0.013 vs +0.01 random) and did nothing: conceptor steering peaked at
  0.23%. Minimum-distortion "causal-metric" steering collapsed to its Euclidean twin (cos > 0.999) because the target
  was rank 1: "no specificity" (`arch/handoffs/steering_operators_report_2026-09-03.md` L240–249, L335–340).
- *At the time:* no curved code, and "the linear localization was not blind to a curved code". **Stands, but moot.**
  The model expressed 0–6% of the effect, so there was little to localize.

**P2. Reach-Wall geometry map, JEPA-WM MetaWorld, 5 Sep, 150 discovery trajectories**
(`js/docs/archive/deferred-threads-2026-09-05/snapshot/Geometry Map Experiment.md` L79–82, L193–200;
`js/scripts/geometry_map/__pycache__/screen_emergence_geometry.*.pyc`).
- (e)+(b) *The PEZ protocol applied to a world model.* Realised XZ motion direction was weak in the image encoder
  (peak CV R² 0.046) and rose to R² 0.573 at predictor block 3, where magnitude also peaked (R² 0.812). Orthogonal
  probes left "useful realized-direction signal after removing eight directions" and fell below threshold after
  ten. Direction was ~75° from the progress and magnitude subspaces and 72° from wall distance. kNN probes "did not
  beat the linear screen".
- (c) Action patching was monotone in all 48 snapshots at dose 0.5: block 2 transferred 0.450 of the intended
  change vs 0.020 for an orientation sham, and block 3 transferred 0.490 vs 0.002.
- *At the time:* "rules out a single-vector coordinate model", so a small multi-direction subspace was the first
  steering candidate. **Never superseded, never followed up.** The thread was deferred the same day, and the
  artefacts are not in the local folder.

**P3. Action-path curvature and learned charts, Push-T, 5–7 Sep** (`js/docs/CURVED_ACTION_RESPONSE_FOLLOWUP.md`;
docstrings in `js/scripts/geometry_map/__pycache__/`).
- (a) Full-spatial action-response paths were bent. Curved interpolation improved causal interchange fidelity over
  endpoint-line reparameterisation, **but** a cubic downstream comparison lost to the near-chord comparator on 3 of
  4 states. Conditioning on actual history improved prediction *and* increased bending. The sample was 4 reused
  states and 32 correlated paths, with targets "already appeared elsewhere in exploration".
- (c) Tried, with no result files locally (they lived in `archive/2026-09-07-workspace/`, absent here):
  - diffusion-map intrinsic charts vs equal-rank PCA, on a "within-state action-direction holdout, NOT
    episode-generalization" (`learn_intrinsic_chart`);
  - a PCA-64 natural-spline path adapted from Goodfire A.3/A.5/A.6 that "preserve[s] the off-subspace residual"
    (`run_density_geometry_pilot`);
  - PCA-4 coordinates with an RBF-64 decoder, and chart controls "leav[ing] the enclosing PCA64 complement
    unchanged" (`native_chart_control`);
  - a mean/radius decomposition of bending, "not proof that residuals live on spheres" (`decompose_action_curvature`).
- (b) A coordinate-transform null recorded that "a frozen linear world readout plus known frame conversion is a
  competing explanation; no new nonlinear network mechanism is required" for an RBF readout gain
  (`coordinate_transform_null`).
- Earlier, a full-spatial correction cut forecast MSE by 26.83% (visual) and 49.35% (proprio) yet worsened two action
  choices (`js/Rank Edit.md`).
- *At the time:* "observed curvature does not establish useful steering." **Stands.** It was kept only as a
  candidate question.

**P4. Registered action-response geometry, 5 tasks × FP32/BF16, 7–8 Sep** (`js/docs/ACTION_GEOMETRY.md` protocol;
`js/docs/PATHWAY_GEOMETRY.md`; `js/findings.md` L82, L262).
- (a) *Design.* One action direction, offsets ±0.05 and ±0.1, and the omitted centre reconstructed from four
  anchors. Four arms:
  - linear, weights 1/4 each;
  - "cubic", weights [−1/6, 2/3, 2/3, −1/6], which equals a quadratic fit at the centre;
  - **projected cubic**, the cubic estimate projected onto the endpoint chord;
  - **reflected curvature**, 2·projection − cubic, i.e. the same bend with the opposite sign.

  The last two separate "speed along a line from a correctly oriented curved deviation" (`js/docs/EXPERIMENT_PLAN.md`).
- (d) Raw reconstruction advantage of cubic: +93.96% to +99.99% in FP32, −35.1% to −46.8% in BF16. Linear-to-cubic
  MSE ratio on Wall and PointMaze: 3,967 and 7,020 in FP32; in BF16, cubic error was 43% and 42% higher.
- (c) All four arms were "inconclusive" on all five tasks. H6 forecast effects were ≤ 0.026% of native MSE, with
  mixed signs.
- (d) Dose effects: requested dose was 7.44 in BF16 vs 0.0066 in FP32. Only 49–51% of FP32 cubic windows passed the
  per-window dose check, against 98% for linear. Dose normalisation *reversed* fidelity: raw error 99.89% lower
  became 16.03% higher after normalising.
- *At the time:* "a specific diagnostic boundary, not a rejection of manifold steering". **Qualified by P5.**

**P5. Controlled geometry, 64 contexts × 6 blocks, 13 Sep** (`js/docs/CONTROLLED_GEOMETRY.md`).
- (d) No fitted bank and no dose normalisation. Log(cubic/linear) centre error: FP32 −5.40 to −7.89; FP32 rounded to
  BF16 and back −0.098 to +0.010; actual BF16 rollout +0.16 to +0.31. The actual-minus-roundtrip lower bounds are
  ≥ 0.209.
- *At the time:* "a numerical mechanism result … not evidence of a physical manifold". Output rounding explains most
  of the flip but not all of it. **Stands.** Reading: at offsets of ±0.1, the "curvature" is a smooth second-order
  response visible only above the arithmetic noise floor.
- The quoted size varies by source. `js/docs/MECHANISM_HYPOTHESES.md` L297 says "~1,000×", which matches P5
  (e^6.7 ≈ 800×), not P4's +94% on Reach (≈ 17×).

**P6. Fresh-v2 geometry of the delivered edit, 13 Sep, post hoc**
(`js/analysis/out/fresh-v2/representation_geometry/representation_geometry.md`).
- Measures the fitted object, not curvature. The participation ratio matched random; the edit basis contains its
  readout (0° angles); verdict "mixed: neither picture cleanly".
- `js/docs/MECHANISM_HYPOTHESES.md` L371–376: "The fresh panel adds nothing about curvature … no manifold claim either way."

**What feeds Part 2 here**

1. **Global-mean blindness (P1).** Averaged over angles, a direction code has mean ≈ 0, the ring's centre. A single
   mean-difference steering vector is therefore the wrong baseline. Steering must be target-conditioned: the Part 1
   least squares per θ\*, or the spline.
2. **The circular chart (P1) is a ready population-level model.** Fit X ≈ μ + A[cos θ, sin θ] on train and decode
   atan2. This matches the paper's per-neuron GLM (C.7) and should agree with Goodfire's atan2(PC2, PC1). The
   local-vs-global projection test answers whether the speed axis rotates with θ (cone vs cylinder).
3. **Planted positive control (P1)** before any null about rings or curvature.
4. **A reflected-curvature arm (P4).** A spline bent the wrong way, with the same endpoints, arc length and dose, is
   the sharpest matched control for spline steering. Add a projected (chord) arm with the spline's spacing.
5. **Noise floor (P5).** The curvature gain must beat within-value clip scatter and BF16 rounding of `meanpool`.
6. **Matched delivered dose (P4).** Normalisation reversed a fidelity result once; compare spline and line per clip at equal ‖Δ‖.
7. **A readout gain is not curvature (P3).** Nonlinear decoding of θ can come from a known frame conversion
   (Cartesian (vx, vy) through atan2). This is the Cartesian-vs-polar question in §8.
8. **A prior estimate (P2).** In a JEPA-WM predictor, motion direction needed ~10 orthogonal directions and sat ~75°
   from magnitude. The paper's 40–136 is for a video encoder. Compare, do not import.
9. **Unseen-donor interchange (P3, `run_action_path_interchange`).** Patch real activations of clips at a held-out
   θ into recipients, and compare with the spline point at that θ. This is the held-out transplant design.

## 7. Representational geometry and steering: applicable findings

| Topic | What they found (source) | Verdict | Use here |
|---|---|---|---|
| Linear vs curved, precision, dose | See §6 (P1–P5) | APPLIES | Leave-one-out cubic vs line on centroids, in FP32 and BF16-rounded, at matched delivered dose, with a planted-ring control |
| Dense vs sparse | "mixed: neither picture cleanly … a small number of directions with distributed support"; 149–252 of 256 effective patches (`representation_geometry.md`) | ADAPT | Report both: rank (INLP K) and spatial spread (disk-pool vs mean-pool) |
| Rank | The rank count was set partly by calibration (PR matched random) (§3) | APPLIES | INLP K is only a claim against the random-removal band |
| Transport | Cross-checkpoint angles ~85–90° vs a random 89.64°; adjacent-layer subspaces 80–89°; "An orthogonal rotation alone is not a contribution" (`representation_geometry.md`; `LOOP1_SUMMARY.md`; `WORLD_MODEL_SONAR_PLAN.md` L217) | ADAPT | Transport across *datasets*: a direction spline built on the direction set, used on speed-set clips |
| Matched random | "Norm matching does not match effect"; calibrated-random vs orientation-only are "different null hypotheses"; one realization "cannot estimate between-basis variability"; m draws give resolution 1/(m+1); the refined control was "byte-identical across all task banks" (`js/matched random control.md` §3–7; `representation_geometry.md` L54) | APPLIES | Two kinds of null, ≥ 20 draws each; never one shared seed |
| Null too easy | An isotropic null is wrong where the real edits lie in a subspace (rank ≤ 20 of 400); "match the empirical mean and covariance, randomly rotate real vectors, or permute" (`js/docs/PLANNING_CONTROL_LITERATURE.md` L15; `root-plans/LRH.md` L17) | APPLIES | Add a covariance-matched random curve (random smooth path through PCA-64) beside the isotropic one |
| Support confound | Goodfire's linear arm replaces the whole residual; its manifold arm keeps the off-PCA residual: "support changes alongside geometry … motivate matched-support baselines" (`PLANNING_CONTROL_LITERATURE.md` L28; `refs/steering_paper.txt` L2717–2724) | APPLIES, top priority | Spline and line edit the same subspace and keep the same residual |
| Dose | Planned but never run: a monotone dose–response along a real interpolation path (`root-plans/JEPA rep geometry.md` §2.5). "Do not normalize every edit to the boundary" (`WORLD_MODEL_SONAR_PLAN.md` L395) | APPLIES | K waypoints; report readout angle *and* radius at each |
| Off-target | Min-distortion steering collapsed to its Euclidean twin (cos > 0.999): "no specificity" (`handoffs/steering_operators_report_2026-09-03.md`). "Collateral effects do not prove superposition" (`WORLD_MODEL_SONAR_PLAN.md` L334) | APPLIES | Steer direction and read speed (and vice versa). First measure the direction–speed principal angles against k/d |
| Saturated readout | First-action hashes changed 96/96 under learned *and* random edits (`forecast_decision_outcome.md` L20–23). The public docs omit the random half | APPLIES | The nearest-centroid "class flip" readout saturates at large doses. Report it with the random arm |
| Trusted readouts | Paired outcome; the model's own forward on a counterfactual input (R). Distrusted: hashes, a proxy MSE that isn't the objective, nHSIC/CKA, significance without magnitude ("recovery 0.000 … reported as an artifact"), coefficient norm as OOD (`INTERVENTION_MECHANISM_AUDIT.md` §4) | APPLIES | Trust: a disjoint evaluation probe plus later-layer agreement with real clips at the target value |
| Semantic/manifold bar | "predict a specified physical variable on held-out examples … preserving appropriate others"; manifold claims need "a fitted chart/metric and an on-manifold versus matched linear comparison" (`js/paper/research_synthesis.md` §3) | APPLIES | This is the bar for the Part 2 headline |

## 8. Variables beyond the three (from the physics paper; our data supports each)

Each row can run on stored activations unless marked. Line numbers refer to `refs/physics_paper.txt`.

| Variable / test | Paper basis | What it would show here | Cost | Part |
|---|---|---|---|---|
| Cartesian (vx, vy) vs polar (speed, sin θ, cos θ): onset layers | Table 1 "Polar factorization dominates" (L81); Fig. 2b–c (L257–260); only vx and ax are plotted | If (vx, vy) comes up before θ, direction's "emergence" may be the normalisation v/‖v‖, which weakens Table 1. Needs constant-speed clips (750 in the direction set, all of the speed set) | minutes | P1 support |
| (ax, ay) vs \|a\|, and the rest-start confound | §5.2 (L264–275); A.1.2: the acceleration set is "initialized at rest" (L684) | In the paper's data, acceleration ≡ mean speed ≡ displacement too, so "no velocity intermediate" is untested there and here. Say so as a critique of the paper | none (slide) | P1 |
| Direction × speed / motion type: ring vs cone vs cylinder | §7.1 ring (L409–420); Goodfire cylinder task with ghost points (`steering_paper.txt` L2640–2647, L3255–3257) | Ring radius vs speed on the speed set (24 of 64 angles per speed) and on velocity vs acceleration clips. A radius that depends on speed makes the manifold 2-D: fit a thin-plate spline over (θ, speed) | minutes | P2 core |
| Speed tuning shape | C.8: quadratic GLM, "preferred speeds at intermediate values" (L1128–1150) | Bump-tuned units imply a *curved* speed centroid path. Do not assume speed is straight; test leave-one-out and log vs linear knot spacing | minutes | P2 |
| Start position / retinotopy | C.5: local → global at the emergence zone; half-frame generalisation (L985–1012); Table 1 "object-centric slots" (L88–91) | Train the direction probe on start x < 0, test on x > 0, per layer, mean-pool vs disk-pool. This is also a held-out *context* for Part 2 steering | minutes | P1 support, P2 |
| Direction vs speed subspace angles | C.4 method and k/d baseline (L891–925). Table 3 (L944–968) reports **only motion vs IntPhys**, never direction vs speed | Whether steering direction can leave speed intact. Random expectation for k = 40 of 1,024 is ≈ 3.9% | minutes | P1→P2 bridge |
| The sin/cos sawtooth | §7.2, Fig. 4c (L377–380); C.10, Fig. 23 (L1178) | With 64 angles, check whether consecutive INLP probes are rotated ~90° in the (s, c) response plane. If not, the sawtooth is a 2-output QR artefact | minutes | P1 |
| Direction dimension, stated three ways | 40–50 (L402); 14–136 (L1217); 66–136 in Table 3, with a flat 400 at layers 20–23 | Report K against the random band; do not aim for any of the paper's figures | none | P1 |
| Held-out evaluation probe fit in-sample | C.12: 103 test clips, R² = 0.99 in d = 1,024 (L1245) | Our evaluation probe needs an out-of-fold R², or the "held-out probe" is not validated | minutes | P1 |
| Temporal locality | C.6 Table 4: temporal attention ablation hurts direction, spatial barely does (L1060–1085) | `timepool` gives a cheap proxy: direction from 2 tubelets vs 8 | minutes | P1 support (optional) |
| Resolution | C.6: 224², 196 patches, 1,568 tokens (L1020–1021) | We use 256² → 16 × 16. Compare by layer fraction only | none | P1 note |

## 9. What not to spend time on (public list, plus the archive stop list)

Attention-distance maps (`docs/PILOT_MECHANISMS.md`); search-entropy measures; output-interaction factorials;
learned-operator search; symmetric four-point curvature tests; stimuli the researcher invents; n = 4 reused states.
From the archive: "Jacobian sonar, bilinear, higher-order, curved-manifold … relational transport" as main-line
instruments; "running geometry before a closed-loop outcome exists"; "three-way-interaction estimands as the first
target"; "protocol versions faster than results" (`xie_lens_2026-09-03.md`). All APPLY. Our "closed-loop outcome" is
Part 1's steering readout, which must exist before Part 2 geometry.

## 10. Ideas for the talk (verdicts)

- The precision flip as the clean negative. APPLIES: the contiguous-arc spline-vs-line test in FP32 and BF16-rounded.
- "Match now, diverge later" (`docs/figures/lcfm_context_lifetime.png`). APPLIES: plot later-layer agreement vs depth
  after a steer.
- The decision-margin certificate. ADAPT: ‖Δ‖ below half the gap to the nearest other centroid means the class
  cannot flip, which predicts null small steers.
- A random control that wins, shown honestly, with per-clip churn (21 rescued, 21 lost on fresh Reach;
  `js/README.md` L204). APPLIES.
- Development vs fresh. APPLIES as a methods slide: +6.25 pp became 0.00 pp, which is why the test split is read once.
