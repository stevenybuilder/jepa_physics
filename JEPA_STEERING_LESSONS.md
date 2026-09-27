# Lessons from jepa_steering for the V-JEPA physics take-home

Source: github.com/stevenybuilder/jepa_steering (commit 8151517, 14 Sep 2026), mined 27 Sep 2026. The value is in its
12 `reports/*_CORRECTION.md` and `*_AUDIT.md` post-mortems, not its code: it has no ridge-probe, INLP, circular-metric
or subspace-steering code, so everything in Part 1 is written fresh. Where a lesson is already in `spec.md` the
section is given.

## Pitfalls, most severe first

| # | What went wrong there | How it would show up here | Rule (spec §) |
|---|---|---|---|
| 1 | Push-T was split by rollout ID, so all 185 initial-state families sat in fit, dev and holdout at once; no confirmation data could be saved (`PUSHT_LINEAGE_CORRECTION.md`) | clips that share a generator seed or start position landing in both train and test | every clip gets a group id; split by group; `validate_split` raises if a group spans two splits (§3) |
| 2 | 160 of 12,600 MetaWorld rows were byte-identical and counted as independent (`METAWORLD_LINEAGE_CORRECTION.md`) | duplicate clips inflate n and leak across the split | sha256 the decoded frames before writing the split; merge duplicates; report the count (§3) |
| 3 | In FP32 a cubic fit beat a linear one at every block; in BF16 linear won. Rounding only the stored outputs to BF16 took a log-ratio from −5.4 to −0.09 (`CONTROLLED_GEOMETRY.md`, `COMPUTE.md`) | INLP's K, QR bases and small steering residuals are all rounding-sensitive | fp32 forward, TF32 off for matmul and cuDNN; `meanpool` stored float32; CPU-vs-GPU tolerance fixed before the run; torch version on the box recorded (§2) |
| 4 | A wrapper silently differed from upstream (`shuffle=True` vs False; a YAML resize overridden by the loader) (`POINTMAZE_TRAINING_SAMPLER_CORRECTION.md`, `DROID_BASELINE_RECONCILIATION_AUDIT.md`) | our deliberate no-resize/no-crop preprocessing hides a real bug, or gets mistaken for one | the smoke test asserts the intended deviation from the HF processor and asserts normalisation constants and token count (§2, §8) |
| 5 | 5 of 286 goal images had wrong hashes from an off-by-one RGB render; whole RNG streams were rerun rather than patching episodes (`METAWORLD_STIMULUS_REPAIR.md`) | activations silently out of step with the videos they came from | bind the frame hash into every activation file; re-extraction must reproduce the same bytes (§2) |
| 6 | One frozen random direction per task; it beat the learned edit post hoc on Reach (`MATCHED_RANDOM_CONTROL_AUDIT_20260911.md`) | "steering works" with no matched random comparison | random subspace of the same rank and per-clip ‖Δ‖, 10 seeds, read by the same evaluation probe (§6.4); 10-seed random-removal band for INLP (§6.3) |
| 7 | Dose matched on average but not per window; cubic arms passed the per-window check 49% of the time (`PATHWAY_GEOMETRY.md`) | mean norm drift looks fine while individual clips blow up | log requested vs delivered ‖Δ‖ per clip, never just the mean (§5.3) |
| 8 | "1,152 rows, not 1,152 contexts"; the bootstrap unit is the family (`CONTROLLED_GEOMETRY.md`, `AUTHOR_VALIDATION_AUDIT.md`) | CIs computed over ~1,500 clips when the independent unit is the label value | bootstrap by label-value cluster; a minimum-detectable-effect at n = 64 (or 48/16) values, not clips (§4) |
| 9 | The "best observed edit" was labelled post hoc; a checker asserts the disclosure text exists (`scripts/check_public_results.py`) | the CV-chosen layer or INLP stopping round presented as a result | label them as selections; test is read once (§3) |
| 10 | A published 48.2 vs their 51.1 was never used as an effect; only a concurrent native baseline counted (`DROID_BASELINE_RECONCILIATION_AUDIT.md`) | comparing our R² to the paper's numbers as if it were evidence | paper numbers are a qualitative reference only; every contrast uses our own controls (§0, §6.2) |

## Practices to copy

- Deterministic split by hash order, independent of input order: `src/offline_study/core/protocol.py::study_split`.
- Manifest guard for duplicate ids and groups crossing splits: `protocol.py::validate_manifest`.
- Content hashing of tensors (dtype, shape, bytes): `src/offline_study/data/inventory.py::_update_tensor_digest`.
- Fail fast on degenerate cases (no finite norm, too few samples, shape mismatch): `fitting/operator_fit.py::_unit`.
- A checker that asserts every README/slide number equals the JSON behind it: `scripts/check_public_results.py::check`.
  Ours: `check_results.py` (§8).
- Zero-dose identity and hook-vs-native byte parity tests before any intervention: `tests/unit/interventions/`.
  Ours: N = 0 steering returns x unchanged; hook output matches native forward (§8).
- Protocol JSON committed before analysis: `paper/data/controlled_geometry_analysis_protocol.json`. Ours: `splits/split_v1.json` first commit.
- Undefined stays undefined: never epsilon-fill or drop an undefined value inside an aggregate. Ours: disk-pool NaN + flag when the disk is out of frame.
- Separate endpoints: "an improvement at one does not establish the next" (`docs/MECHANISMS.md`). Ours: decodable → steerable at the same layer → propagated through later layers are three claims.

## Liftable helpers

`analysis/mechanism/common.py` (`paired_bootstrap`, `holm`, `write_json`, `sha256`); `representation_geometry.py`
(`random_subspace`, `principal_angles_deg`, `participation_ratio`); `operator_fit.py::orthogonal_random_control`;
`core/protocol.py` (`study_split`, `validate_manifest`, drop its dataset whitelist).

## Where spec.md was corrected because of this review

- float16 storage of the main activations → float32 for `meanpool`.
- No run precision stated → fp32, TF32 off, tolerance and torch version recorded.
- "Processor parity" test → "intended processor deviation" assertion.
- No duplicate/family audit before the split → frame hashing and `validate_split`.
- Steering had norm drift and off-target readouts but no matched random control → added.

## Scientific findings and what they mean for Part 2

Paths below are relative to the jepa_steering repo root unless they start with `~/`. That project steered the
action-conditioned predictor of JEPA-WM (6 blocks, 256 patches × 400 features, edit at block 3 / imagined step 3) and
scored forecasts with a CEM planner. The one claim not in the repo (the TRM note) comes from
`~/.claude/projects/-Users-stevenyang/memory/rep-geometry-transcoder-2026-09-06-state.md`.

### A. Representational geometry

| Finding | Numbers | Source |
|---|---|---|
| The response to one action coordinate is strongly curved in FP32 and the curvature disappears in BF16 | Mean ln(cubic/linear omitted-centre MSE) was −5.40 (block 0) to −7.78 (block 5) on Reach in FP32. Rounding the FP32 outputs to BF16 gave −0.088 to +0.010; a real BF16 rollout gave +0.16 to +0.30 (the straight line won at every block, 64 contexts) | `docs/CONTROLLED_GEOMETRY.md` |
| Same flip on 5 tasks, with fitted edits | Cubic's raw reconstruction advantage was +93.96% to +99.99% in FP32 and −35.1% to −46.8% in BF16 | `docs/PATHWAY_GEOMETRY.md` |
| Curvature didn't matter for behaviour | Cubic-vs-linear edits changed step-6 forecast error by at most 0.026% of native MSE, with mixed signs. "No geometry arm qualifies on any task" | `docs/PATHWAY_GEOMETRY.md`, `reports/CORRECTED_OFFLINE_RESULTS.md` |
| The curvature test was weaker than its name | The weights [−1/6, 2/3, 2/3, −1/6] on four symmetric anchors equal a quadratic fit at the centre, so the test "cannot identify third-order dynamics", "a dense manifold, or model-native physical axes" | `docs/PATHWAY_GEOMETRY.md`, `docs/MECHANISMS.md` |
| Dose confound across precisions | The requested edit size for Reach was 7.4422 in BF16 and 0.0065586 in FP32. Only 49.02% of FP32 cubic windows passed the per-window dose check, against 98.08% for linear | `docs/PATHWAY_GEOMETRY.md` |
| Dimensionality: one direction is not enough | Rank 4 and rank 8 met the rank-selection rule; rank 1 did not | `reports/CORRECTED_OFFLINE_RESULTS.md` |
| Inside the rank-4 subspace, only about 1.4 to 3 directions carry the edit | Participation ratio (tr M)²/tr(M²) of the delivered coefficients: 2.23 Reach, 1.37 Reach-Wall, 3.07 PointMaze, 1.92 Wall | `docs/RESULTS.md`; code in `analysis/mechanism/representation_geometry.py::participation_ratio` |
| The edit is mostly one constant shift | 98.855% (Reach) and 99.615% (Reach-Wall) of coefficient energy is shared across candidates. The shared part alone reproduces the relative cost change at 0.983 and 0.998 | `docs/MECHANISMS.md`, `docs/PILOT_MECHANISMS.md` |
| A constant shift still changes rankings | ‖z+d−g‖² − ‖z−g‖² = 2d·(z−g) + ‖d‖², and the first term differs across candidates | `docs/MECHANISMS.md` |
| Staying inside the input's own subspace mattered only a little | The action encoder is a rank-20 map into 400 dimensions. Random edits inside its range disrupted rankings more than edits outside it at blocks 1 and 4, by about 0.0021–0.0025 Spearman. "Affine encoder-range membership is not membership in a physical or global activation manifold" | `docs/ACTION_COUNTERFACTUAL.md` |

Density and off-manifold energy were never measured. The repo computes principal angles and participation ratios
only for fitted bases (`representation_geometry.py`), and no output from that script is committed.

**Verdict:** the local curvature is real in FP32 and has almost no effect on behaviour. Both "dense nonlinear
manifold" and "linear" are left explicitly unsupported.

### B. Steering

- **What was tried:** a learned rank-four edit at the block-3 output, applied to all 256 patches and all features;
  visual-plus-action coupling with each part scaled by 1/√2; a rank-one sweep over every block; linear, cubic,
  projected and reflected geometry edits; action-conditioning swaps (`docs/METHODS.md`,
  `reports/CORRECTED_OFFLINE_RESULTS.md`). No partial spatial support (subsets of patches) passed the advancement
  rule, so the edit needs all patches (`reports/CORRECTED_OFFLINE_RESULTS.md`).
- **What worked (forecast error only):** the learned rank-four edit cut step-6 proprioceptive MSE by 2.36% on Reach
  and 2.19% on Reach-Wall. Joint coupling plus rank 4 gave 4.824% [3.668, 5.980] (`docs/RESULTS.md`,
  `reports/CORRECTED_OFFLINE_RESULTS.md`).
- **Layers:** in the rank-one sweep, earlier blocks corrected more: 3.026% at block 0 falling to 0.475% at block 5
  on Reach (BF16). Push-T didn't reproduce this (`docs/MECHANISMS.md`). Action-history sensitivity peaked at block 1
  (`docs/ACTION_COUNTERFACTUAL.md`). The chosen block 3 was "not retroactively justified".
- **Better forecasts didn't change decisions.** The first choice stayed the same in 191 of 192 states. The rule
  "the edit's cost spread is smaller than the winner's lead" certified 184 of them in advance (`README.md`,
  `docs/MECHANISMS.md`). Physical 15-step prefix endpoints: all 8 intervals include 0
  (`docs/PLANNED_PREFIX_REPLAY.md`). Protected success on fresh scenarios: all 48 contrasts include 0. On Reach,
  21 scenarios were rescued and 21 lost, so both arms stayed at 52/96 (`docs/RESULTS.md`).
- **Learned was never better than random.** Adaptive search changed under both the learned and the random edit, and
  all 6 learned-minus-random intervals include 0 (`docs/CEM_EXPANSION.md`). In development, the random coupling
  control scored 60.42% on Reach, the highest observed rate (`reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md`).
- **Dose–response:** only the precision flip (A) and the per-block decline were measured. There was no graded dose
  curve against a behavioural endpoint.
- **Transport:** there was no cross-task operator; every task got its own fit (`docs/METHODS.md`). Push-T joint
  edits made forecasts worse (−0.638%) (`reports/CORRECTED_OFFLINE_RESULTS.md`).
- **"Readout-subspace fix works on nav, not contact" (TRM, arXiv 2605.22164):** the fix in that paper leaves the
  world model alone and changes the planner's terminal cost to use the subspace a position probe reads. That took a
  navigation task from 7% to 96.7% success (a shuffled control gave 0%), but only moved Push-T, a contact task, from
  40% to 52.7%. Nothing published edits the model's internals for contact tasks. The sister project's version of
  this idea (ablating everything outside the probe span) gained +1.16 pp at n=4 (memory file above). In short, a
  probe subspace helps when the target is literally position; it helps much less when the physics goes beyond it.

### C. Methods that transfer to a frozen video encoder plus predictor

| Method (repo source) | What it is | Analogue for V-JEPA 2 |
|---|---|---|
| Read-site sweep with all layers reported (`docs/MECHANISMS.md` layer map) | Edit every block and show the full grid, never one chosen block | Yes: steer at every depth in the emergence zone, report all of them |
| Matched controls (`docs/METHODS.md`) | A random subspace of the same rank, support and dose with its own calibration; isotropic vs in-range random | Yes: a random curve with matched length through PCA-64, and a spline through centroids with shuffled labels |
| Decodable → used ladder (`docs/MECHANISMS.md`) | Four separate endpoints: activation reconstruction, forecast, candidate score, closed-loop success. "An improvement at one does not establish the next" | Partial: readout at the steer layer → readout at later layers after propagation → predictor forecast readout. There is no planner |
| Action counterfactual (`docs/ACTION_COUNTERFACTUAL.md`) | R = 1 − D(patch, cf)/D(unmodified, cf) against the model's own forward on a changed input. Patching both occurrences reproduced it byte-exactly; patching one gave R ≈ 0.50 | **Yes, the strongest one:** compare against real clips with the target value (the Goodfire centroid, or a re-rendered clip if the generator is available), at the steer layer and every later layer |
| Planned-prefix replay (`docs/PLANNED_PREFIX_REPLAY.md`) | Execute the plans each model chose, from identical resets | None: there are no actions |
| History ranking (`docs/LCFM_HISTORY_RANKING.md`) | Spearman, top-10 overlap and winner agreement with the counterfactual reference; an edit made at one time step only lost 17 of 32 winners | Yes: the nearest of the 64 centroids to the steered activation is the "winner". Also edit all 8 time steps vs some of them |
| Decision-margin certificate (`docs/MECHANISMS.md`) | If the edit's cost spread is smaller than the winner's lead, the choice cannot change | Yes: if ‖Δ‖ is less than half the gap to the nearest other centroid, the class cannot flip. This predicts which small steers are null before running them |

### D. What this means for Part 2 (manifold steering on V-JEPA)

1. **Where a spline should win.** Curvature was large locally and irrelevant at small doses (A), so a spline should
   beat a straight line only when the steer goes far. For direction, a straight path between far angles cuts
   through the middle of the circle. Report the win as a function of |θ − θ\*|, which the 64 directions allow
   (spec §5.3). For speed, expect little difference inside the training range and a gap only at far or held-out
   values. Acceleration here is the same variable as displacement (spec §1), so an acceleration spline can't be told
   apart from a speed/distance spline. Say so rather than claiming a separate manifold.
2. **Cheap checks before fitting splines, using the stored float32 `meanpool`:**
   - leave-one-value-out centroid reconstruction, linear vs cubic, at held-out knots (asymmetric, unlike the repo's
     test, which could only see second-order structure);
   - the same test with the stored activations rounded to BF16 (a one-line check against A's flip);
   - the ratio of within-value scatter to the curvature signal;
   - participation ratio of the 64 centroids vs of the within-value noise;
   - principal angles between the INLP basis, the centroid PCA plane, and the speed and direction subspaces;
   - the off-manifold energy of the Part 1 linear steer. If it already stays near the centroid curve, a spline has
     little room to win.
3. **Shared vs clip-specific share of each Δ.** Split each steer into its mean over clips and each clip's residual,
   as in `docs/MECHANISMS.md`. At 99% shared, a spline step and a linear step are nearly the same edit.
4. **What Goodfire's metrics catch:** steps that leave the data (off-manifold energy; compare the isotropic vs
   in-range results) and uneven step sizes (isometry).
5. **What they miss, each seen in the repo:**
   - curvature that is a precision artefact (A);
   - an edit that matches at the steer layer but is undone later, when unedited tokens or later layers re-read the
     original value. This is the second-read divergence: 42 of 100 and 59 of 100 winners changed
     (`docs/LCFM_REPLICATION.md`);
   - a probe fit on the same centroids that scores an on-manifold steer well by construction;
   - no win over a random curve (B);
   - dose matched on average but not per clip (A).
6. **Which behavioural readout to trust:** agreement with the model's own forward on a real clip that has the target
   value, measured at the steer layer and every later layer, plus an evaluation probe fit on disjoint clips at
   held-out values (spec §6.1). A probe readout at the same layer is the weakest rung; forecast gains of 2% did not
   move decisions (B). The sister project found that the native latent-L2 goal cost misranks physically better
   actions even with oracle futures (memory file above), so V-JEPA 2 predictor latents are a readout, not ground
   truth.

### E. What not to spend time on

- Attention-distance maps. They are "a distance map, not a causal map", and heads showed 1.106–9.521 patches of
  spread (`docs/PILOT_MECHANISMS.md`).
- Search-entropy or proposal-drift measurements. All intervals span 0 (`docs/CEM_EXPANSION.md`,
  `docs/PILOT_MECHANISMS.md`).
- Output-interaction factorials. The effects were ≤0.08% of MSE on the MetaWorld tasks (`docs/PATHWAY_GEOMETRY.md`).
- Searching over learned operators. Learned never beat random (B); the sister project got 29 null configs at n=4
  (`~/.claude/projects/-Users-stevenyang/memory/postmortem-2026-09-06-three-projects.md`).
- Symmetric 4-point "cubic" tests (A).
- Stimuli the researcher invents, and n=4 reused states (postmortem memory).
- Hash-binding every artefact. The repo's docs are mostly provenance text; keep what the first pass listed and stop
  there.

### F. Ideas worth stealing for the talk

- **The clean negative:** "numerical precision can reverse a geometry result" (`README.md`,
  `docs/figures/geometry_control_story.png`). The Part 2 version: run the held-out-knot spline-vs-line test in FP32
  and in BF16-rounded activations, and show whether the winner holds.
- **Match now, diverge later:** `docs/figures/lcfm_context_lifetime.png`. The Part 2 version plots the
  counterfactual score R against depth after a steer, for one-shot vs all-token edits.
- **Certificate CDF:** perturbation divided by margin, with a line at 1 (`docs/figures/decision_margin_story.png`).
  It explains nulls in advance.
- **A random control that wins, shown honestly** (Reach-Wall's best arm was randomised coupling; `README.md`).
  Also show the per-clip churn behind a net-zero result (21 rescues and 21 losses; `docs/RESULTS.md`).
- **The shared-shift identity** 2d·(z−g) + ‖d‖² as a slide on why a "boring" constant steer still moves readouts
  (`docs/MECHANISMS.md`).
- **Figures that show every layer and arm, with no chosen layer** (`docs/figures/layer_mechanism_*`).
