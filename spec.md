# Spec: V-JEPA physics take-home, Part 1

Sources: `vjepa-physics-takehome-4E00/{README,DATA}.md`; Joseph et al. 2026, `physics paper.pdf` (= arXiv 2602.07050v1),
read in full on 27 Sep 2026: §3.2, §5, §7, App. A.1.2, B, C.5, C.11, C.12. The paper released no code. Learnings from
`MEMJEPA_LESSONS.md` are folded in where they change a decision, and marked (MJ).

## 0. What Part 1 must produce

The README asks for the paper's *main experimental progression* on the supplied data, reproducing methodology and
qualitative findings, not numbers:

| Step | Paper | Qualitative finding to check |
|---|---|---|
| 1. Layer-wise probing | Linear probes on mean-pooled space-time tokens at every layer (§3.2, App. B), for speed, direction, acceleration magnitude, and the Cartesian pairs (Fig. 2) | Speed and acceleration decodable from early layers; direction only from ~1/3 depth (the Physics Emergence Zone); performance peaks mid-network and falls toward the output |
| 2. Iterative nullspace probing | Orthogonal probe sequence at one layer, repeated until chance (App. C.11, Figs. 4c, 22, 23) | Direction needs tens of dimensions with a sawtooth curve; speed needs fewer and has no sawtooth |
| 3. Multi-probe subspace steering | QR basis from the K probes; least-squares steering; held-out evaluation probe (App. C.12, Fig. 24) | 1–5 probes barely move the readout; ~20 probes reach the target; MAE-to-true rises as MAE-to-target falls |

Three figures, one per step, carry the Part 1 talk. Everything else is support.

Hard rules: encoder frozen; `data/` read-only; every derived file lives outside it; document hook point, preprocessing
and pooling; no fitted component ever touches the final test clips; every slide number comes from a results file.

## 1. Data (checked 27 Sep 2026)

| Dataset | Clips | Target | What else varies |
|---|---:|---|---|
| direction | 1,500 | θ, 64 values, 23–24 clips each | half constant-speed (1–7 m/s), half accelerating from rest (2–10 m/s²); start position in [−2, 2]² |
| speed | 1,536 | 64 speeds 0.25–4.0, 24 clips each | 64 directions, fully crossed; start in [−1.2, 1.2]² |
| acceleration | 1,536 | 64 accelerations 0.25–10.0, 24 each | 64 directions, fully crossed; starts at rest |

Differences from the paper's data: 64 directions instead of 8 (the circle can be drawn directly), labels in m/s,
one merged direction set that mixes both motion types. Two things to state on slides rather than fix:
- acceleration clips all start at rest, so acceleration, mean speed and total displacement are one variable here.
  The paper's acceleration set has the same confound (sphere "initialized at rest", App. A.1.2), so this limits the
  paper's own "acceleration decodable without a velocity intermediate" claim, not only our reproduction. The 750
  accelerating direction-set clips also start from rest and cannot separate it either;
- fast clips can leave the frame. A data-QA step records the fraction of frames with the disk visible per clip.

## 2. Model, extraction, pooling

- `facebook/vjepa2-vitl-fpc64-256` via `transformers==4.56.2` (installed; pinned). 24 blocks, width 1024, patch 16,
  tubelet 2. 16 frames at 256² give 8 × 16 × 16 = 2,048 tokens (verified on CPU, 10.6 s per clip).
- Preprocessing: decode all 16 frames with PyAV, scale by 1/255, ImageNet mean/std. No resize or crop (the default
  processor resizes to 292 and crops to 256, which clips borders and rescales pixel speed by 1.14×).
- Hook point: `output_hidden_states=True` returns 25 states in this version: index 0 the embedding, 1–23 the residual
  stream after blocks 1–23, and 24 = final LayerNorm of block 24's output (checked in the source). A forward hook on
  block 24 captures its raw output. Stored: 26 points = embedding, blocks 1–24, final post-LN. Plots use layer
  fraction, as in the paper. Smoke test asserts `len(hidden_states) == 25`, the token count, the normalisation constants against the HF
  processor, and that our tensor differs from the processor's output only by the skipped resize/crop (the deviation is
  asserted as intended, not hidden).
- Precision: fp32 forward with TF32 disabled for matmul and cuDNN (the tubelet embed is a Conv3d, which PyTorch runs
  in TF32 by default on Ampere+). jepa_steering lesson: TF32/BF16 flipped a geometry result there. `meanpool` is stored
  float32; the two 8-step pools float16. CPU-vs-GPU check tolerance: originally max |Δ| < 1e-3 on 8 clips. Measured 27 Sep
  (`artifacts/gpu_session1.json`): local-CPU vs box-GPU 3.9e-3, but local-CPU vs box-CPU 4.6e-3 and same-GPU batch 8
  vs 16 6.0e-3, all in blocks 17–24 where activations reach ~150 (worst relative error 2.4e-4). The absolute bar was
  below fp32 accumulation noise on any device. Revised rule: per-layer relative error max|Δ|/max|x| < 1e-3, and the
  GPU-vs-GPU batch gap must not exceed the CPU-vs-CPU gap by more than 2×. Torch version on the box recorded.
- Pooling, one row per manifest `id`, each file carrying the sha256 of the decoded frames it came from:
  - `meanpool` [N, 26, 1024] float32: mean over all 2,048 tokens. **This is the paper's representation; every Part 1 result
    uses it.**
  - `timepool` [N, 26, 8, 1024]: mean over the 256 patches per time step.
  - `diskpool` [N, 26, 8, 1024] + `diskmask` [N, 8, 16, 16]: mean over only the patches the disk covers (red channel > 128:
    the disk is orange, RGB ≈ (226, 113, 43), not blue as DATA.md says; max-pooled to the patch grid over both frames
    of a tubelet); NaN + flag where the disk is out of frame.
    This is a cheap stand-in for the paper's per-patch analysis (App. C.5) and is what tells us whether pooling hides
    the signal.
  - Full tokens are not stored (~19 GB per layer). Anything that writes into the model re-runs the encoder from the
    videos on the GPU.
- Same pass also extracts `meanpool` from a randomly initialised ViT-L (fixed seed) as a control. A pixel baseline
  (frames downsampled to 32², plus frame differences, PCA-256 fit on train) is computed locally.
- Total download ≈ 4.8 GB.

## 3. Split: the paper's protocol (`splits/split_v1.json`, written and committed before the first probe number)

The paper (App. B, C.11, C.12) uses random clip splits with a fixed seed: 5-fold cross-validation with the model
chosen on validation performance for the layer curves; 80/20 train/test for the orthogonal probe sequence; 70/30 for
steering, with the evaluation probe fit on the test-set activations and the steering subspace built only on train.
DATA.md adds one requirement: probe fitting, layer selection and nullspace construction must not use the final test
examples. The paper's protocol already satisfies it (selection happens on the CV validation folds).

We use the same protocol with one random split per dataset so every experiment reads the same clips:

- **`train` (80%) / `test` (20%)**, random clips, seed 0, stratified by label value so each of the 64 values appears
  in both. Duplicate clips (by frame hash) are merged before splitting; `validate_split` raises if a clip is in both.
- **5 stratified folds inside `train`** are the validation data: ridge α, layer choice and the INLP stopping round
  are all decided on fold-held-out scores. Fold mean ± SD is what the paper plots.
- **`test` is read once per experiment.** For steering it plays the paper's 30% role: the evaluation probe is fit on
  `test` activations, the steering subspace on `train`, and `test` clips are steered (App. C.12, steps 1–5). The
  ratio is 80/20 rather than 70/30; with ~1,500 clips instead of 343 that is immaterial, and it keeps one split.

Held-out *label values* (steering to a value the subspace never saw, splines evaluated at unseen knots) are not part
of the paper's method. They belong to Part 2, where the README asks us to decide "what constitutes a meaningful
held-out steering evaluation"; see §6 item 1. Part 1 does not depend on them.

"Held-out" is used in three senses and every result says which it meets: excluded from *fitting*; excluded from
*development decisions* (layer, K, knot count, spline type, α); *untouched* until the final read. All Part 2 design
choices are made on the train folds. `test` is read once for Part 1 and once for Part 2's final table; anything
chosen after a test read is labelled exploratory. (jepa_steering's development gain of +6.25 pp was 0.00 pp on
fresh scenarios; the three senses had been conflated.)

## 4. Metrics

Direction: probe outputs (sin θ, cos θ); report R² on the pair and circular MAE in degrees (chance ≈ 90°), plus the
readout radius ‖(ŝ, ĉ)‖ per clip (ridge readouts shrink below 1; atan2 of a near-zero readout is noise, and a straight
path between opposite angles passes through radius 0, so dose curves need the radius beside the angle). Speed and
acceleration: R² and MAE in physical units. Cartesian (vx, vy), (ax, ay) as in Fig. 2b. Uncertainty: mean ± SD over
the 5 folds, as the paper reports; 95% bootstrap CIs on test where a difference is claimed.

## 5. The three experiments

### 5.1 Layer-wise probing (paper §3.2, App. B, Fig. 2)
- Probe: `f(h) = Wh + b` on train-standardised `meanpool`. Fit as closed-form ridge with α chosen by 5-fold CV
  inside `train` over 13 log-spaced values, as the paper selects on validation performance (App. B). Same model class as the paper's Adam + weight-decay probe, deterministic and fast.
  Check once per variable at one layer that the paper's Adam recipe (App. B grid) matches ridge within the CI.
- Grid: 5 targets × 26 layers on V-JEPA. Controls (shuffled labels, pixel baseline, random-init ViT-L) are §6 item 2.
- Also run the same grid on `diskpool` (mean over the 8 time steps). Mean-pool vs disk-pool is the pooling check.
- Output: `fig1_layer_curves` (fold mean ± SD vs layer fraction, per variable, direction split by motion type in a
  panel; controls overlaid where run); `fig1b_direction_circle` ((ŝ, ĉ) on test coloured by θ at three layers); a
  table of test scores at each variable's chosen layer.
- Layer choice for steps 2–3: the layer with the best CV score per variable, expected in the emergence zone.
- Cheap support runs on stored activations, after the core grid (all Part 1 support, minutes each):
  (a) **Cartesian vs polar onset**: onset layer of (vx, vy) vs (sin θ, cos θ) on constant-speed clips, same
  availability rule and bootstrap CI. The paper asserts polar dominates (Table 1) but shows only vx and ax curves;
  if (vx, vy) comes up earlier, direction's "emergence" may be the normalisation v/‖v‖.
  (b) **Direction transfer** (`fig1c`): direction probe fit on the direction set, evaluated on speed-set clips
  (speeds 0.25–4 m/s, starts in [−1.2, 1.2]², partly outside the direction set's 1–7 m/s and [−2, 2]²) and on
  acceleration-set clips. Held-out context at no extraction cost.
  (c) **Spatial generalisation** (paper C.5 / Fig. 18b stand-in): direction probe trained on start x < 0, tested on
  x > 0 and the reverse, per layer, on `meanpool` and `diskpool`.
- Note for Q&A: the paper's attention analysis used 224² input (14 × 14 patches, 1,568 tokens); ours is 256² without
  crop (16 × 16, 2,048 tokens). Layer fraction is comparable, patch-level counts are not.

### 5.2 Iterative nullspace probing (paper App. C.11)
- At the chosen layer per variable, and every layer for the dimensionality-vs-depth plot (Fig. 22).
- Loop, k = 1, 2, …: fit probe P_k on X⁽ᵏ⁾ (train); score on the CV folds with the same accumulated projection; Q_k = QR of
  W_kᵀ (2 columns for direction, 1 for scalars); X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ − X⁽ᵏ⁾Q_kQ_kᵀ. α fixed at the step-1 value. Everything in
  the one standardised coordinate system; nullspaces here are not orthogonal in raw space, so nothing mixes the two.
- Stop (paper): direction R² < 0.1 or circular MAE > 80°; speed and acceleration R² < 0.05 or MAE > 90% of the
  predict-the-mean baseline. Dimensionality = 2K (direction) or K (scalars). Also report K at the R² < 0.3 threshold
  Fig. 22 uses. The paper itself gives three figures for direction's dimension (40–50 in §7.2, 14–136 in C.11,
  66–136 in Table 3 with a flat 400 at layers 20–23 that looks like a cap); we report our K against the random band
  and do not aim for any of them.
- One control (MJ): random orthonormal subspaces of matched rank projected out, 10 seeds. "Tens of dimensions" means
  something only against that band. Note on the slide that n ≈ 1,200 train clips vs d = 1024, so K is a fold-scored ridge count.
- Output: `fig2_inlp` (score vs round for direction and speed, random band behind; does the sawtooth appear for
  direction and not speed?), `fig2b_dim_vs_layer`; bases saved for step 3.

### 5.3 Multi-probe subspace steering (paper App. C.12)
- Basis V = QR([W_1ᵀ … W_Kᵀ]). Steer: c = Vᵀx, x⊥ = x − Vc; least squares for c\* so the first N probes all read
  the target; x\* = Vc\* + x⊥. Sweep N = 1…K.
- **Protocol (the paper's, App. C.12):** steering probes from `train` (the orthogonal probe sequence until R² < 0.1);
  evaluation probe fit on `test` activations only; steer `test` clips toward θ\* = 90°, and also toward every one of
  the 64 directions; plot MAE-to-target and MAE-to-true vs N (paper: 82.9° → 11.9° at N = 20, single probe > 50°).
  Report the result as a function of the angular shift |θ − θ\*| as well, which 64 directions make possible.
  The evaluation probe's α is chosen by CV within `test` and its R² is reported **out-of-fold** (the paper's 0.99 is
  in-sample from 103 clips in d = 1024; ours is ~300 clips, still n < d).
  Steering target: the paper's unit vector (sin θ\*, cos θ\*) is the reproduction; also run a radius-matched target
  (mean readout of real train clips at θ\*), since least squares toward a unit target pushes activations beyond the
  data when ridge readouts have radius < 1.
- Same procedure for speed and acceleration with scalar probes and unseen magnitudes (Part 2's baseline).
- Random nulls, two kinds, ≥ 20 draws each: (i) a random orthonormal basis of the same rank with its own
  least-squares solve (calibrated random); (ii) the learned c\* applied through a random basis Q with ‖Qc‖ = ‖Vc‖ per
  clip (orientation only). Report the empirical rank; one draw cannot estimate between-basis variability.
- Logged for free: per-clip norm drift ‖x\*‖/‖x‖ and the speed readout before/after steering direction (§6 item 4).
- Output: `fig3_steering` (the paper's Fig. 24 on our data), `fig3b_shift_heatmap`.

## 6. Additions beyond the paper, ranked, each with its trigger

Run only after §5 is complete. None is needed for Part 1. The reasoning behind Part 2 (why steer at all, what
spline steering is, why direction's circularity is the test case, what jepa_steering's nonlinearity results do and
do not transfer, the five-rung evidence ladder and the decision tree) is in `PART2_RATIONALE.md`; the judgment
items are in `nonobvious_components.md` §4–10.

1. **Held-out label values + strict evaluation probe (Part 2's evaluation design).** Fit the evaluation probe on
   clips disjoint from both the knot/subspace clips and the steered clips, and steer toward values never used as
   knots. Three held-out designs, reported separately: (a) **scattered**, every 4th value held out (a local
   interpolation check; the largest remaining gap is 11.25°, where a chord and a cubic nearly agree, so this cannot
   separate spline from line); (b) **contiguous**, one 45° arc of direction (8 values) and one interior block of 8
   speed and 8 acceleration values held out; the spline-vs-line claim rests on this one; (c) **extrapolation**, the
   top 8 speeds and accelerations, labelled as such. A held-out value is also the one place where an *endpoint*
   readout can separate the methods: there is no centroid to land on, the line's target is a chord point and the
   spline's is the curve point, and they differ by the sagitta, which grows with arc length. Precedent for held-out
   values: Kantamneni & Tegmark 2025, arXiv 2502.00873, App. C.2. Running the Part 1 subspace method under the same
   design gives Part 2 its like-for-like baseline. Trigger: start of Part 2.
   - **Matched support.** Spline and line arms edit the same subspace (centroid PCA-k plane or the Part 1 basis V)
     and both keep the clip's off-subspace residual. Goodfire's own linear baseline replaces the whole activation
     with a chord point while its manifold arm keeps the residual, so its comparison mixes "residual kept vs erased"
     with "curved vs straight". Run Goodfire's version once, labelled as such.
   - **Two curvature controls** with the same endpoints, dose and waypoint count: *reflected* (2·chord − spline, the
     bend flipped) and *projected* (the chord traversed with the spline's spacing). A random curve tests bending at
     all; the reflected curve tests bending the right way.
   - **Held-out context:** steer speed-set clips (unseen speeds and start range) with a direction spline built on
     the direction set.
2. **Controls on the layer curves:** shuffled-label probe, pixel baseline (32² frames + frame differences, PCA-256),
   random-init ViT-L, last-frame-only, and **time-reversed clips** (same occupancy, opposite direction; GPU session 2,
   trivial to extract). In a clean fixed-camera scene raw pixels can match V-JEPA 2 (CALIPER, arXiv 2609.08250), so
   selectivity must be shown, not assumed. Trigger: any variable whose curve is high from the embedding layer onward (a disk on black is
   nearly pixel-decodable). The random-init extraction is cheap, so it is done in GPU session 1 regardless.
3. **Random-removal band on the INLP curve** (random subspaces of matched rank, 10 seeds). Trigger: before claiming
   "tens of dimensions" on a slide.
4. **Matched random steering control, per-clip norm drift, off-target readout** (§5.3 readouts). Trigger: same.
5. **Propagation, closure depth, and the predictor as behavioural readout** (GPU session 2, one batched session).
   (a) Broadcast the pooled Δ to every token at layer L, run the remaining blocks, read direction at every later layer
   with probes refit per read layer: heatmap steer-layer × read-layer (Othello washout / closure-depth precedent,
   arXiv 2609.15980). (b) **The simulator dial**: render counterfactual twin clips (same start and speed, direction
   θ\*; validate the renderer against 20 supplied clips first), encode the *context frames only* (HF `VJEPA2Model`
   applies `context_mask` after encoding, so encoding all 16 frames lets the context see the future), apply the edit,
   run the predictor, and score the predicted future tokens with a probe-free recovery
   R = ⟨ẑ_edit − ẑ_src, ẑ_tgt − ẑ_src⟩ / ‖ẑ_tgt − ẑ_src‖² against the twin's real future, plus future-direction error.
   Controls: unsteered predictor preserves direction; random matched rank and norm; shuffled target; ‖Δ‖ relative to
   the median natural change. This is the readout Sonia's own essay asks for ("steer that velocity representation and
   observe corresponding counterfactual changes in future predictions"); nobody has judged an edit to any JEPA by its
   predictor (lit_review.md §2). ~200 carriers × 6 arms × 4 layers, under an hour on a 4090. Trigger: after Part 1
   picks the layer and Part 2's cheap checks pass.
6. **Attentive probe** (GPU session 2, tokens kept on the box only). Trigger: the mean-pool direction curve shows no
   clear rise, or disk-pool beats mean-pool by more than the CI. The paper's Fig. 18a says the mean-pool curve rises
   gradually, so a gradual curve is the expected reproduction.
7. **Direction-vs-speed and direction-vs-acceleration subspace angles** (paper C.4 method: principal angles,
   projection overlap ‖Q_AᵀQ_B‖²_F / dim B, random expectation k_A/d), per layer, from the INLP bases. The paper's
   Table 3 only compares motion with its possible-vs-impossible task, so direction vs speed is unmeasured there.
   This decides whether steering direction can leave speed untouched (the off-target readout). Trigger: before any
   off-target claim; minutes on stored bases.

8. **"How many dimensions is direction?" as four estimands, one figure** (stored activations, minutes). The
   iterative-erasure count is not affine-invariant: Jin et al., arXiv 2608.10566, cite the paper's C.11 by name and
   show that on a synthetic ring a feature shear turns K = 1 into K ≈ 6 while a whitened metric returns rank 2 every
   time. Per layer, direction vs speed: (a) the literal K vs the random-removal band (the reproduction); (b) whitened,
   residualised K (Jin Alg. 1, rank(I − T)); (c) fresh ridge and MLP R² after a rank-2 LEACE erasure (arXiv
   2306.03819); (d) split-half DFT spectrum of the 64 direction centroids (harmonics k = 0…8) plus participation ratio.
   Readings: whitened K ≈ 1 with LEACE-2 at chance means the literal K is conditioning; mass at k ≥ 2 means a curved
   ring with harmonics (her "harmonic basis" remark); LEACE-2 failing means redundant copies. Control: a planted ring
   of known harmonic content and copy count pushed through all four estimators. Labelled as an extra; does not replace
   the paper's count. (LEACE was on the not-doing list; it is reinstated only in this role.)
9. **"Fewer probes": low-rank steering bake-off at matched edit norm** (stored activations). Her blog asks "whether
   more efficient steering methods can recover the same control with fewer probes". Arms: probe-QR least squares
   (N = 1…K, the reproduction); centroid transport x + μ(θ\*) − μ(θ); rotation of the k = 1 ring plane by Δθ (rank 2,
   norm-preserving); periodic spline in PCA-k (Part 2); nearest-centroid snap (floor). Figure: held-out-probe angle
   error vs edit rank at matched ‖Δ‖. Claim under test: rank 2–4 matches the ~40-dim edit. Controls: random subspace
   of equal rank with its own solve; learned coefficients on a random basis; sham edit; an MLP evaluator on disjoint
   clips (a linear evaluator shares filter geometry with probe-QR; precedent Marks & Tegmark 2310.06824, AxBench
   2501.17148 where probe AUROC 0.94 gave steering 0.10 vs mean-difference 0.24).
Not doing: regularisation sweeps of K, MLP layer curves, per-patch heatmaps, attention ablations, neuron
tuning. Backup slide only.

## 7. Pre-registered expectations (MJ: written before any run)

| Step | Expected | Falsified if | Slide either way |
|---|---|---|---|
| 1 | speed/accel high early; direction rises at ~1/3 depth; peaks mid, falls late | direction already high at the embedding, or nothing beats the pixel baseline | curves with controls |
| 2 | direction 2K in the tens, sawtooth; speed K smaller, no sawtooth | INLP K inside the random band | INLP vs random band |
| 3 | MAE-to-target falls with N, single probe fails, MAE-to-true rises | flat in N, or single probe already reaches target | Fig. 24 on our data |

Outcomes on real data (27 Sep, `results/p1a_*.json`, `p1b_*.json`, `p2_geometry_*.json`; kept next to the expectations,
not edited into them):
- Step 1: **falsified for onset.** All three variables are linearly decodable from block 1 (CV R² 0.88–0.98 at
  point 1, ≈ 0 at the embedding point), no emergence zone, no late decline. A random-init ViT-L reaches 0.85 (direction),
  0.92 (speed), 0.91 (acceleration) and 0.99 on (vx, vy) at block 1 and stays flat. Raw pixels with a linear probe do
  not (0.15 / 0.44 / 0.35); the disk-trajectory floor makes every Cartesian target exactly linear but cannot read a
  magnitude (R² 0.00). Reading: one attention block with rotary positions turns displacement into linear velocity and
  a norm-like magnitude, trained or not; V-JEPA's training adds precision with depth (direction 10.7° → 3.0°, speed
  MAE 0.11 → 0.07), not availability. Start position predicts direction at chance (R² 0.00), so this is not a data
  confound.
- Step 2 (direction): **"tens of dimensions" reproduces at the peak layer** (point 22: K = 23, 46 dims; K = 20 at the
  Fig. 22 threshold; paper §7.2 says 40–50). At the onset layer (point 2) the code is far more redundant (K = 104,
  208 dims). **No sawtooth** at either layer on R², MAE or within-15° accuracy (fraction of rises 0.0, drop
  autocorrelation 0.92–0.98); consecutive probe planes are 6–8° apart, i.e. successive probes are weakening copies
  of one (sin, cos) mixture, not alternating sin/cos features. Random removal of the same number of dimensions leaves
  R² unchanged (0.99 → 0.99).
- Part 2 pre-checks (direction, points 8/12/22): a ring exists (supervised circular chart fits the 64 centroids at
  3.5–7° MAE, radius 3.2–7.3, growing with depth); at point 12 its plane is 6° from the top-2 PC plane. Local
  curvature is below centroid noise at knot spacings ≤ 45° (sagitta 0.04–0.4 vs noise ≈ 1.5) and detectable at 90°.
  Speed at point 12 is a straight, evenly spaced line (knot-spacing R² 0.999 linear, participation ratio 1.65): the
  pre-registered negative.

## 7b. Deviations from the paper, in one table (Part 1)

| Item | Paper | Here | Why | Effect |
|---|---|---|---|---|
| Probe fit | Adam + weight decay, grid in App. B | closed-form ridge, α by 5-fold CV over 13 values | same model class, deterministic | checked once per variable at one layer (`scripts/check_probe_recipe.py`) |
| Split | 70/30, C.11 stopping on an 80/20 test split | one 80/20 clip split, stratified by value; stopping on fold-mean CV, test read once | ~1,500 clips vs 343; one split | none expected |
| Input | 224², 14 × 14 patches, 1,568 tokens | 256² no crop, 16 × 16, 2,048 tokens | supplied clips are 256² | layer fractions comparable, patch counts not |
| Hidden states | 24 residual points | 26 points: embedding, blocks 1–24 raw, final LN | hook on block 24 | post-LN point plotted separately |
| Data | 8 directions, separate speed/accel sets | 64 directions; direction set mixes constant and accelerating clips | supplied data | direction reported per motion type |
| INLP layer | layer 8 (emergence) | onset layer from the availability rule AND CV-peak layer | Fig. 22 shows the count roughly doubling late | both reported |
| Steering basis | V from raw stacked W_1..W_N (C.12 Eq. 8) | V from residualised probe weights | probes stored orthogonal to earlier directions, so identical up to float error | tested for equality |
| Extras | none | radius readout, radius-matched target, two random nulls, out-of-fold eval-probe R², §6 items | labelled extras in JSON and figures | never in the core figure |

## 8. Code and execution

```
src/wm/{data,extract,splits,probes,inlp,steer,figures}.py   scripts/{extract,run_step1,run_step2,run_step3,make_figures}.py
splits/split_v1.json   artifacts/ (gitignored)   results/*.json   figures/
```
Environment: the existing `.venv` (Python 3.11, torch 2.2.2, transformers 4.56.2, PyAV) plus scikit-learn, scipy,
matplotlib; pinned in `requirements.txt`. Tests: 8-clip CPU smoke test (token count, shapes, intended processor deviation), split validation,
circular-MAE wrap-around, INLP on synthetic data with a planted rank, zero-dose steering identity (N = 0 or target =
true value returns x unchanged). `check_results.py` asserts every number in the slides/README against `results/*.json`
(copied from jepa_steering `scripts/check_public_results.py`). Liftable helpers from that repo:
`analysis/mechanism/common.py` (`paired_bootstrap`, `write_json`, `sha256`), `representation_geometry.py`
(`random_subspace`, `principal_angles_deg`), `core/protocol.py` (`study_split`, `validate_manifest`). No probe, INLP
or steering code exists there; those are written fresh.

Order and cost:
1. Local: packages, data loading + QA, split file, extraction script, 8-clip smoke test. Nothing is debugged on a paid box.
2. GPU session 1, one 4090-class Vast box: extraction, ~30–45 min including setup, under $1. Verify shapes and re-check
   a few clips against the CPU values, download 4.5 GB, then release the box on the user's say-so.
3. Local: steps 1 → 2 → 3, `make_figures`.
4. GPU session 2, later and batched with Part 2's needs: §6 items 1–2.

Part 1 gets ~25% of the total effort.
