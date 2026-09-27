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
- acceleration clips all start at rest, so acceleration, mean speed and total displacement are one variable here;
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
  float32; the two 8-step pools float16. CPU-vs-GPU check tolerance fixed in advance: max |Δ| < 1e-3 on 8 clips, torch
  version on the box recorded.
- Pooling, one row per manifest `id`, each file carrying the sha256 of the decoded frames it came from:
  - `meanpool` [N, 26, 1024] float32: mean over all 2,048 tokens. **This is the paper's representation; every Part 1 result
    uses it.**
  - `timepool` [N, 26, 8, 1024]: mean over the 256 patches per time step.
  - `diskpool` [N, 26, 8, 1024] + `diskmask` [N, 8, 16, 16]: mean over only the patches the disk covers (blue-channel
    threshold, max-pooled to the patch grid over both frames of a tubelet); NaN + flag where the disk is out of frame.
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

## 4. Metrics

Direction: probe outputs (sin θ, cos θ); report R² on the pair and circular MAE in degrees (chance ≈ 90°). Speed and
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

### 5.2 Iterative nullspace probing (paper App. C.11)
- At the chosen layer per variable, and every layer for the dimensionality-vs-depth plot (Fig. 22).
- Loop, k = 1, 2, …: fit probe P_k on X⁽ᵏ⁾ (train); score on the CV folds with the same accumulated projection; Q_k = QR of
  W_kᵀ (2 columns for direction, 1 for scalars); X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ − X⁽ᵏ⁾Q_kQ_kᵀ. α fixed at the step-1 value. Everything in
  the one standardised coordinate system; nullspaces here are not orthogonal in raw space, so nothing mixes the two.
- Stop (paper): direction R² < 0.1 or circular MAE > 80°; speed and acceleration R² < 0.05 or MAE > 90% of the
  predict-the-mean baseline. Dimensionality = 2K (direction) or K (scalars). Also report K at the R² < 0.3 threshold
  Fig. 22 uses.
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
- Same procedure for speed and acceleration with scalar probes and unseen magnitudes (Part 2's baseline).
- Logged for free: per-clip norm drift ‖x\*‖/‖x‖ and the speed readout before/after steering direction (§6 item 4).
- Output: `fig3_steering` (the paper's Fig. 24 on our data), `fig3b_shift_heatmap`.

## 6. Additions beyond the paper, ranked, each with its trigger

Run only after §5 is complete. None is needed for Part 1.

1. **Held-out label values + strict evaluation probe (Part 2's evaluation design).** Refit the steering subspace on
   clips from 48 of the 64 values (every 4th value held out, interior for speed/acceleration), fit the evaluation probe
   on clips disjoint from both the subspace clips and the steered clips, and steer toward the 16 unseen values. This
   is what a "meaningful held-out steering evaluation" means for splines (a spline through all 64 centroids evaluated
   on the same 64 is interpolation by construction; precedent: Kantamneni & Tegmark 2025, arXiv 2502.00873, App.
   C.2). Running the Part 1 subspace method under the same design gives Part 2 its like-for-like baseline. Trigger:
   start of Part 2.
2. **Controls on the layer curves:** shuffled-label probe, pixel baseline (32² frames + frame differences, PCA-256),
   random-init ViT-L. Trigger: any variable whose curve is high from the embedding layer onward (a disk on black is
   nearly pixel-decodable). The random-init extraction is cheap, so it is done in GPU session 1 regardless.
3. **Random-removal band on the INLP curve** (random subspaces of matched rank, 10 seeds). Trigger: before claiming
   "tens of dimensions" on a slide.
4. **Matched random steering control, per-clip norm drift, off-target readout** (§5.3 readouts). Trigger: same.
5. **Propagation test** (GPU session 2): add Δ to every token at the steering layer, run the remaining blocks, read
   direction at later layers. Trigger: if time; it is the readout Part 2's predictor comparison needs anyway.
6. **Attentive probe** (GPU session 2, tokens kept on the box only). Trigger: the mean-pool direction curve shows no
   clear rise, or disk-pool beats mean-pool by more than the CI. The paper's Fig. 18a says the mean-pool curve rises
   gradually, so a gradual curve is the expected reproduction.

Not doing: LEACE, regularisation sweeps of K, MLP layer curves, per-patch heatmaps, attention ablations, neuron
tuning. Backup slide only.

## 7. Pre-registered expectations (MJ: written before any run)

| Step | Expected | Falsified if | Slide either way |
|---|---|---|---|
| 1 | speed/accel high early; direction rises at ~1/3 depth; peaks mid, falls late | direction already high at the embedding, or nothing beats the pixel baseline | curves with controls |
| 2 | direction 2K in the tens, sawtooth; speed K smaller, no sawtooth | INLP K inside the random band | INLP vs random band |
| 3 | MAE-to-target falls with N, single probe fails, MAE-to-true rises | flat in N, or single probe already reaches target | Fig. 24 on our data |

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
