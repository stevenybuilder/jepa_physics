# V-JEPA 2 physics take-home: report

Every number below comes from a file in `results/` (including `results/split70`, `results/split70_p2`,
`results/arcs`, `results/stimuli`) or `artifacts/gpu_session1.json`, named in a footnote or in the table's source line.
Paper claims are stated qualitatively and attributed to the paper.

## 1. Summary

The README asks for a small-scale reproduction of Joseph et al. (arXiv 2602.07050): layer-wise probes, iterative
nullspace probing, and multi-probe subspace steering for direction, speed and acceleration in the frozen V-JEPA 2
ViT-L/16 encoder. It also asks for an open-ended Goodfire spline-steering extension (arXiv 2605.05115) that takes care
over the circular structure of direction and over what counts as a held-out steering evaluation. **Step 1 does not
reproduce on the supplied clips.** All three variables are linearly decodable from block 1, and there is no emergence
zone and no late decline. A random-init ViT-L and random nonlinear features of the disk trajectory decode them nearly
as well, so on this stimulus availability comes from architecture plus a clean input. Training adds precision. On a
harder rendered set (textured floor, shading, smaller disk; 392 clips, 8 directions) the direction onset moves from
point 2 to point 5 and V-JEPA's block 1 falls below the random-init network's: a partial recovery of the emergence
zone. VideoMAE (pixel reconstruction) matches V-JEPA 2 on all three variables with the same onsets. **Step 2
reproduces in part.** Every variable needs tens of probes at the paper's layer, far outside a random-removal band.
Measured four ways, that count reflects anisotropy: a whitened erasure needs one probe at every point; the code is
rank-2 linear plus a second harmonic and a nonlinear residual (§3.2). Speed does not need fewer probes than direction,
and the ridge curves have no sawtooth. Under the paper's literal Adam recipe both variables' curves are jagged and K
roughly doubles, with no direction-specific sawtooth. **Step 3 reproduces in shape** (one probe fails, a few probes
reach the target, MAE-to-true rises), with fewer probes than the paper needs. However, a random orthonormal basis of
rank 2K matches the learned one up to N≈5, the same curve appears in an untrained network, and at the onset layer the
learned basis never beats the random one. Rerunning Parts 1 and 2 at the paper's literal 70/30 split changes no
qualitative verdict. **Part 2.** Direction lies on a ring that is recovered without labels at mid depth. At held-out
direction values the spline path stays on the ring and the straight path cuts through it: the minimum readout radius
is higher for the spline by +0.26 ± 0.05 at point 12 and +0.24 ± 0.07 at point 22 (mean ± SD over 16 runs covering 15
distinct held-out 45° arcs), and it is higher on every arc. The endpoint error ties at point 12 (+0.08° ± 1.47) and
slightly favours the line at point 22 (+1.26° ± 3.54; 3 of 16 runs flagged "negative_endpoint"). Readouts that did not
build the edit (nearest-real-clip agreement, an MLP on disjoint clips) do not separate the two. Speed and acceleration
are straight lines, and there the spline adds nothing. **Beyond the steered layer (GPU session 2).** Edits at points ≤
12 wash out within a few blocks. Point-22 edits survive to the output with little target specificity (shuffled targets
are at most 39° away). The predictor test is uninformative, not negative: the unedited forecast reads its own clip's
direction 61° off and the real counterfactual context recovers only R = 0.0185 of the real future change, so it could
not have seen an edit. The Part 2 advantage is geometric at the steered layer; downstream use is still open.

## 2. Setup

- **Model.** `facebook/vjepa2-vitl-fpc64-256` (transformers 4.56.2), frozen. Input is 16 frames at 256², decoded
  with PyAV, divided by 255 and normalised with ImageNet mean/std. There is no resize or crop (`src/wm/extract.py`),
  giving 8×16×16 = 2,048 tokens. Forward passes run in fp32 with TF32 off for matmul and cuDNN (flags recorded as
  false[^gpu]). There are 26 hidden-state points: the embedding, blocks 1–24 (the raw output of block 24, captured by
  a hook), and the final LayerNorm. The paper's "layer L" is our point L+1, so the paper's layer 8 is point 9.
- **Representation.** `meanpool`, the mean over all 2,048 tokens, is used for every Part 1 result, as in the paper.
  `diskpool` (tokens the disk covers) and `timepool` (per time step) are used for controls.
- **Data.** direction 1,500 clips (64 angles, a mix of constant-velocity and accelerating-from-rest clips); speed
  1,536 (64 speeds × 64 directions); acceleration 1,536. The README says this directly: *"The supplied dataset is
  deliberately smaller and simpler than the datasets in the paper. The aim is to reproduce the methodology and
  qualitative findings, not the paper's exact numerical results."* The stimulus is one orange disk on a flat
  background with a fixed camera. Acceleration clips all start at rest, so acceleration, mean speed and displacement
  are one variable here. The paper's acceleration set has the same confound.
- **Split.** `splits/split_v1.json` is one random 80/20 split per dataset, stratified by value (direction 1,200/300,
  speed and acceleration 1,228/308[^s1][^steer]). Ridge α, layer choice and INLP stopping use 5 folds inside train.
  Test is read once per experiment. Every results JSON carries a provenance block (split sha256, git commit, dirty
  flag).
- **Probe.** Closed-form ridge on train-standardised features, with α chosen by 5-fold CV over 13 log-spaced values
  (`src/wm/probes.py`, `ALPHAS = np.logspace(-2, 4, 13)`). Direction is fit to (sin θ, cos θ) and scored by R², circular
  MAE and readout radius.

**Deviations from the paper** (spec §7b, checked against code where cheap):

| Item | Paper | Here | Effect / check |
|---|---|---|---|
| Probe fit | Adam + weight decay | ridge, α by CV | parity check below: ridge ≥ Adam on all three variables |
| Input | 224², 1,568 tokens | 256², no crop, 2,048 tokens | layer fractions comparable, patch counts not |
| Hidden states | 24 points | 26 (embedding, blocks 1–24, final LN) | paper layer L = our point L+1 |
| Data | 8 directions, separate sets | 64 directions, mixed motion types | direction also reported per motion type |
| Split | 70/30, C.11 stopping on test | 80/20, stopping on fold-mean CV | "paper protocol" also run; Parts 1 and 2 rerun at 70/30, no verdict changes (§3.4) |
| INLP K | probes until test at chance | nested K (held-out folds), paper-protocol K beside it | both below |
| Steering basis length | until R² < 0.1 on train | all-train sequence cut at nested K | length never chosen on test |
| INLP recipe | Adam lr 1e-3, wd 1e-4 | ridge; literal Adam sequence run once at points 8 and 9 | curves jagged and K ≈ 2× under Adam; no direction-specific sawtooth (§3.2) |

**Four places where the paper's text disagrees with itself, and how each was handled**[^ptxt]. (a) The INLP stopping
threshold for direction is R² < 0.1 in C.11 and R² < 0.3 in the Fig. 22 caption: each `p1b` file has both (`K`,
`K_loose`; §3.2). (b) The split is 80/20 in C.11 and 70/30 in C.12: we ran both (§3.4). (c) C.12 fits its evaluation
probe on the test clips it then steers and scores (independent of the steering probes but not of those clips): her
protocol is our §3.3 headline, with a split-half version beside it (12.4° / 17.1° at N = 5 vs 8.7°). (d) The main text
reports < 0.5° with all probe directions, the appendix 11.9° held-out with 20 probes: we report every N.

**Probe-recipe parity** (point = CV-peak layer; pooled out-of-fold R², targets standardised for Adam[^recipe]):
ridge vs Adam (C.11 recipe) is 0.9905 vs 0.9858 for direction, 0.9940 vs 0.9882 for speed and 0.9925 vs 0.9871 for
acceleration. The best coupled-L2 Adam grid cells reach 0.9868, 0.9907 and 0.9887 (AdamW 0.9870, 0.9905, 0.9889).
Every ridge value lies above the Adam CI, so ridge is slightly *better* on all three variables.

## 3. Part 1

### 3.1 Layer-wise probing

**Paper's claim.** Speed and acceleration are decodable early. Direction appears only from about one third of the
depth (the "Physics Emergence Zone"). Performance peaks mid-network and falls toward the output.

**Figure:** `figures/fig1_layer_curves.png` (the direction curve jumps at point 1; both motion types are shown in the
right panel).

| Target | pt 0 | pt 1 | pt 2 | pt 9 | peak (pt) | onset [95% CI] | decline peak→final | CV MAE pt 1 → 9 → peak |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direction (sin, cos) | 0.100 | 0.875 | 0.931 | 0.980 | 0.991 (22) | 2 [2, 2] | 0.0008 | 10.7° → 4.0° → 3.0° |
| speed | −0.002 | 0.983 | 0.983 | 0.988 | 0.994 (19) | 1 [1, 1] | 0.0015 | 0.109 → 0.092 → 0.067 m/s |
| acceleration | −0.019 | 0.977 | 0.975 | 0.982 | 0.992 (21) | 1 [1, 1] | 0.0022 | 0.339 → 0.304 → 0.197 m/s² |
| (vx, vy) | 0.039 | 0.985 | 0.986 | 0.984 | 0.990 (22) | 1 [1, 1] | 0.0007 | |
| (ax, ay) | 0.002 | 0.975 | 0.976 | 0.980 | 0.989 (22) | 1 [1, 1] | 0.0007 | |

Cells are CV R² (5-fold mean). Onset is the first point at ≥ 90% of the maximum, with a 200-draw clip bootstrap for
the CI. Source: `p1a_{var}_meanpool.json`.

**Verdict: falsified for onset.** Direction is at 0.875 after one block, and no variable declines late. Disk-pooling
changes little (direction peak 0.994, onset still 2[^disk]).

**Controls. The non-reproduction is explained by the stimulus, not by a bug.**

| CV R² | direction | speed | accel. | (vx, vy) | (ax, ay) |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2, block 1 | 0.875 | 0.983 | 0.977 | 0.985 | 0.975 |
| V-JEPA 2, peak | 0.991 | 0.994 | 0.992 | 0.990 | 0.989 |
| random-init ViT-L, block 1 | 0.848 | 0.920 | 0.914 | 0.989 | 0.982 |
| shuffled labels, block 1 | −0.014 | −0.014 | −0.012 | −0.007 | −0.010 |
| raw pixels (32² frames + diffs, PCA-256) | 0.150 | 0.438 | 0.346 | 0.833 | 0.878 |
| disk-centroid trajectory, linear (32 numbers) | 0.805 | −0.001 | −0.002 | 1.000 | 1.000 |
| 1,024 random ReLU features of the trajectory | 0.875 | 0.982 | 0.970 | 1.000 | 1.000 |
| same, of the frame differences | 0.993 | 0.999 | 0.996 | 1.000 | 1.000 |

Source: `p1a_baseline_comparison.json` (which cites the per-control files).

Reading. The trajectory is exactly linear in the Cartesian targets, and one nonlinearity turns it into speed and
direction. The random ViT-L's block 1 sits at that random-feature level, and V-JEPA's block 1 is no higher. The random
network stays flat with depth (direction CV MAE 10.9° → 9.5° at its peak), while V-JEPA's improves (10.7° → 3.0°).
Training buys precision, not availability. The selectivity of V-JEPA over the random network at the direction peak
is 0.125 R²[^obj].

**Paper-scale subsample** (train rows only, 10 seeds where random; `figures/fig1f_paperscale.png`[^ps]):

| Condition (direction) | n clips | V-JEPA block 1 → 8 | onset [CI] | random ViT block 1 → 8 | onset |
|---|---:|---|---|---|---|
| full train | 1,200 | 0.875 → 0.973 | 2 [2, 2] | 0.848 → 0.864 | 1 |
| paper's 8 directions | 150 | 0.590 → 0.933 | 5 [4, 5] | 0.664 → 0.699 | 1 |
| 8 directions, direction-grouped folds | 150 | 0.396 → 0.899 | 5 [5, 8] | 0.647 → 0.695 | 1 [1, 2] |
| 150 random clips, 64 directions | 150 | 0.638 → 0.933 | 5 [3, 6] | 0.715 → 0.737 | 1 |
| 240 random clips | 240 | 0.686 → 0.943 | 4 [3, 5] | 0.734 → 0.750 | 1 |

At the paper's sample size V-JEPA's direction onset moves later, to point 4–5, and the rise over blocks 1–5 becomes
selective (the random network stays flat). The cause is sample size: 150 clips over 64 directions behave like 150
clips over 8. The onset still does not reach the paper's one-third depth. Speed stays at onset 1 for V-JEPA in every
condition. The random network's speed onset becomes unstable (direction-grouped folds: 0.294 → 0.468, onset 11 [3,
20])[^pss].

**Rendered stimulus sets** (8 directions × 7 speeds × 7 starts = 392 clips, 313 train / 79 test, direction only;
`figures/stimuli_{paper_layout,hard}_examples.png`)[^stim]. The *paper-layout* set keeps the supplied look. The *hard*
set adds a textured floor, shading and a disk of half the radius.

| Set (direction CV R²) | V-JEPA block 1 | random ViT block 1 | V-JEPA peak (pt) | onset [CI] | random onset |
|---|---:|---:|---|---|---|
| paper layout | 0.822 | 0.851 | 0.992 (22) | 2 [2, 3] | 1 |
| hard | 0.811 | 0.881 | 0.992 (22) | 5 [5, 7] | 1 |

The two sets have the same size and the same 8 directions, so the shift from onset 2 to onset 5 on the hard set comes
from the rendering, not from sample size. On the hard set V-JEPA's block 1 is below the random network's, and V-JEPA
first beats it beyond the paired CI at point 5 (post-hoc selectivity onset). The random network stays flat or declines
(0.881 → 0.861). This is a partial recovery of the emergence zone: point 5 is about a fifth of the depth, not the
paper's third. Caveats: 392 clips, 8 directions, direction only, one render seed.

**Objective axis: VideoMAE** (ViT-L, pixel reconstruction; 224-px input, 1,568 tokens)[^obj]. VideoMAE matches V-JEPA 2
on every variable. Direction: block 1 0.886 vs 0.875, peak 0.992 (point 21) vs 0.991 (22), onset 2 for both. Speed:
peak 0.996 vs 0.994, onset 1. Acceleration: peak 0.996 vs 0.992, onset 1. Nested K at each model's peak is 67 / 109 /
87 for VideoMAE vs 88 / 89 / 67 for V-JEPA 2. Steering reaches the bar with 4 / 8 / 9 probes vs 4 / 7 / 6
(`figures/fig5_objective_axis.png`). Nothing in Part 1 on this stimulus is specific to latent prediction.

### 3.2 Iterative nullspace probing

**Paper's claim.** Direction needs tens of dimensions and its curve has a sawtooth. Speed needs fewer and decays
smoothly. **Figures:** `figures/fig2_inlp.png` (point 9) and `figures/fig2b_dim_vs_layer.png`.

| Point | Direction: nested K [fold range] / K at Fig. 22's R² < 0.3[^fig22] / paper-protocol K | Speed: nested K / paper K | Accel.: nested K / paper K |
|---|---|---|---|
| onset (2; 1 for scalars) | 289 [197–383] / 152 / 395 | 361 [350–385] / 410 | 466 [336–509] / 554 |
| 8 | 40 [33–47] / 21 / 83 | 45 [39–53] / 45 | 48 [44–51] / 50 |
| **9 (paper layer 8)** | **37 [33–42] / 23 / 46** | **39 [36–45] / 45** | **41 [36–44] / 47** |
| peak (22 / 19 / 21) | 88 [69–97] / 37 / 94 | 89 [78–100] / 103 | 67 [61–70] / 74 |
| random-init ViT, pt 9 | 24 [15–31] | 9 [7–10] | 11 [10–12] |

K counts probes, as on the paper's Fig. 22 y-axis; the text's "dimensions" is 2K for direction (point 9: 74). Source: `p1b_{var}_meanpool_L{pt}.json`
and `p1b_*_random_L{pt}.json`.

- **Random-removal band.** Projecting out a random subspace of matched rank (10 seeds) leaves the score unchanged.
  At point 9, removing 92 random dimensions leaves direction CV R² at 0.980, the same as with nothing removed. At
  point 22, removing 194 leaves it at 0.990. "Tens of probes" is therefore a real count, far outside the band.
- **Direction vs speed.** Probe counts are equal (37 vs 39 at point 9, 88 vs 89 at the peaks). Speed needs fewer
  *dimensions* only because its probes are 1-output. The paper's second claim does not reproduce in probe count.
  Early layers hold each variable in hundreds of weak redundant directions (onset rows), which fits the
  random-feature picture from step 1.
- **Sawtooth.** Under ridge there is none. There are no isolated dips at any direction layer under either protocol.
  The lag-1 autocorrelation of per-round drops in R² is positive everywhere (0.42–0.95; a sawtooth gives negative
  values). On within-15° accuracy it is positive under the nested protocol (0.26–0.78) and mixed under the paper
  protocol (−0.24 at point 2, −0.06 at point 8, 0.27 at point 9, 0.19 at point 22).
- **Adam sequence (the C.11 recipe, run literally, targets standardised).** Direction reaches chance after K = 84
  probes (batch 64) or 96 (full batch) at point 9, and 88 or 100 at point 8. Speed takes 95 or 91 at point 9 and 104
  or 102 at point 8. That is about 2× the ridge paper-protocol K at point 9 (46 direction, 45 speed); for direction at
  point 8 it is 1.1–1.2× (ridge 83). Both variables turn jagged. The drop autocorrelation is negative for direction on
  within-15° accuracy (−0.30, −0.32 at point 8; −0.37, −0.34 at point 9) and for speed on R² (−0.44, −0.49; −0.54,
  −0.47). Direction has no isolated dips in any run; speed has 1, 2, 2 and 0. No round failed to train[^adam]. So the
  literal Adam recipe gives a jagged curve and a larger K for both variables, with no direction-specific sawtooth.
  Both are properties of the optimiser, not of the representation.

**How many dimensions, four ways** (train clips only; `figures/fig2c_dims_four_ways.png`)[^dim4], for her open
question (is direction "organized around a harmonic basis rather than a set of independent feature axes"?) and for Jin
et al.'s proof (arXiv 2608.10566) that the literal INLP count is not invariant to invertible reparameterisation.

| | literal K | whitened K | LEACE-2 ridge R² | LEACE-2 MLP R² | centroid power k = 1 / 2 / 3 |
|---|---:|---:|---|---|---|
| direction, pt 9 / 12 / 22 | 37 / 64 / 88 | 1 | 0.980 / 0.984 / 0.991 → −0.001 | 0.70 → 0.37 / 0.72 → 0.41 / 0.87 → 0.82 | 0.72 / 0.25 / 0.005 (pt 9); 0.66 / 0.26 / 0.011 (pt 22) |
| speed, pt 9 (LEACE-1) | 39 | 1 | 0.988 → −0.000 | 0.84 → 0.76 | |
| planted: clean ring / ×30 sheared / + k = 3 harmonic / 3 copies | 1 / 8 / 1 / 1 | 1 | → −0.009 | → −0.24 to −0.31 | 0.995 / 0 / 0; 0.96 / 0 / 0; 0.66 / 0 / 0.32; 0.98 / 0.01 / 0 |

Whitened K is 1 at every point 0–25 for both variables and every ε (0.001–0.1). After LEACE, ridge reads nothing, an
MLP keeps 0.37–0.82, and every planted ring drops below −0.23. In the centroid DFT, k = 2 holds 0.21–0.37 of the
non-constant power from point 8 on (0.08–0.16 at points 2–7) and k ≥ 3 together 0.03–0.11. The controls calibrate each
column: shear inflates literal K (8) but not whitened K, a planted k = 3 harmonic shows up in the DFT (0.32) with
literal K = 1, and copies do not inflate K. So (i) direction is organised on a harmonic basis (k = 1 plus a cos 2θ
term shared by opposite directions), not on tens of independent axes, and (ii) the literal count behaves like Jin et
al.'s sheared circle. "Tens of dimensions" is a conditioning count of an anisotropic rank-2 linear code, not an
intrinsic rank; beyond that code lies a nonlinear residual that grows toward the output.

**Verdict.** The "tens of dimensions" claim reproduces at the paper's layer against a random band. The claims that
speed needs fewer and that direction's curve has a sawtooth do not reproduce under ridge. Under Adam both curves are
jagged, speed's as much as direction's, so the jaggedness tracks the recipe, not the variable.

### 3.3 Multi-probe subspace steering

**Paper's claim (Fig. 24).** One to a few probes barely move the readout, about 20 reach the target, and MAE-to-true
rises as MAE-to-target falls. **Figures:** `figures/fig3_steering_paper.png`, `figures/fig3c_steering_nulls_paper.png`,
`figures/fig3b_shift_heatmap_paper.png`.

Direction, target θ\* = 90°, evaluation probe fit on test (paper protocol; out-of-fold R² 0.967 at point 9, 0.980 at
point 22, 0.860 at point 2):

| N probes | pt 9 MAE-to-target | pt 9 random-basis mean (p) | pt 9 MAE-to-true | pt 22 MAE-to-target (p) | pt 2 MAE-to-target (p) |
|---:|---:|---|---:|---|---|
| 0 | 87.4° | | 2.3° | 87.3° | 86.5° |
| 1 | 78.4° | 77.7° (0.67) | 9.1° | 71.7° (0.52) | 85.7° (0.62) |
| 3 | 25.4° | 30.0° (0.33) | 62.0° | 12.9° (0.29) | 82.1° (0.57) |
| 5 | 8.7° | 12.8° (0.14) | 78.9° | 5.5° (0.29) | 77.3° (0.62) |
| 10 | 3.1° | 9.0° (0.095) | 85.4° | 4.4° (0.14) | 65.0° (0.48) |
| 20 | 2.9° | 14.2° (0.048) | 87.3° | 2.8° (0.048) | 21.4° (0.57) |
| K (37 / 88 / 289) | 2.7° | 81.2° (0.048) | 87.0° | 4.1° (0.048) | 6.9° (0.048) |

p is the empirical rank against 20 random orthonormal bases of the same rank, each with its own least-squares solve
(the floor is 1/21 = 0.048). The orientation-only null (learned coefficients through a random basis) gives p = 0.048
at every N, with a mean of 87.4° at N = 1. Source: `p1c_direction_L{2,9,22}.json`.

- **Null reading.** The learned basis is no better than a random basis until N ≈ 5 at point 9 (p ≥ 0.14), and it
  beats the random basis from N = 10–20. At the onset layer it never beats it for N ≤ 20. The paper's "single probe
  fails, many succeed" is largely what least squares does in *any* subspace of that rank. The learned directions
  become specific only once N is past a handful. Qualification: the random basis has rank 2K = 74 at every N
  (`random_nulls.rank`), not the learned basis's 2N, so "random matches learned up to N ≈ 5" holds for this 74-dim
  null; a rank-matched null is weaker and would favour the learned basis earlier (not stored in `results/`). At 70/30
  the learned basis is at the 1/21 floor at N = 10 for all three variables (80/20: p = 0.095 / 0.14 / 0.095)[^s70].
- **Untrained network.** The same curve appears in the random-init ViT at point 9 (K = 24): 82.9° → 30.7° (N = 5)
  → 8.7° (N = 10) → 5.9° (N = 20)[^p1cr].
- **Norm and radius.** The median ‖x\*‖/‖x‖ is 1.003 at N = 20 and 1.118 at N = K (point 9), and 1.364 at N = K at
  point 22. The readout radius dips to 0.72 at N = 2 and returns to 0.99 by N = 10. The radius-matched target arm
  gives the same MAE curve (`fig3_steering_paper.png`, right).
- **Off-target.** Steering direction and reading speed (speed probe fit on the direction set's constant-velocity
  clips, unsteered MAE 0.17 m/s): the mean absolute change in the speed readout is 0.16 m/s at N = 5, 0.21 at N = 20
  and 0.18 at N = K at point 9. At point 22 it is 2.85 m/s at N = K = 88. Steering speed and reading direction: 2.9°
  (N = 5), 1.5° (N = 20), 3.9° (N = K) at point 9, and 8–30° at point 1.
- **Speed and acceleration** (point 9; target 2.15 m/s and 5.20 m/s²): speed 0.95 → 0.16 (N = 5) → 0.08 m/s (N ≥ 10),
  random-basis p 0.52 / 0.14 / 0.048 at N = 5 / 10 / K. Acceleration 2.48 → 0.40 → 0.23 m/s², p 0.43 / 0.095 / 0.048.
  Norm ratio 0.94–0.95 at N = K.
- **Strict evaluation** (extra: test split in half, evaluation probe fit on one half, the other half steered; point
  9, evaluation OOF R² 0.96): 12.4° / 17.1° at N = 5 and 4.1° / 4.2° at N = 10 (the two directions of the split),
  against 8.7° and 3.1° under the paper protocol. The in-sample evaluation probe flatters small N slightly and makes
  no difference from N = 10[^strict].

**Verdict.** Fig. 24's shape reproduces, reaching the target with 5–10 probes rather than 20. Measured against a
random basis and an untrained network, it is weaker evidence that the probe directions are special than the figure
alone suggests.

### 3.4 The paper's 70/30 split

Part 1 was rerun at the paper's literal 70/30 split (`splits/split_paper70.json`: same seed, stratification and
identical-clip grouping; `figures/split70/`)[^s70]. Every qualitative verdict is unchanged (`verdicts_changed: []`).

| 80/20 → 70/30 (train/test 1,200/300 → 1,050/450 for direction, 1,228/308 → 1,075/461 for the scalars) | direction | speed | acceleration |
|---|---|---|---|
| onset / peak point | 2 / 22 → 2 / 22 | 1 / 19 → 1 / 19 | 1 / 21 → 1 / 19 |
| nested K / paper-protocol K at point 9 | 37 / 46 → 36 / 35 | 39 / 45 → 36 / 40 | 41 / 47 → 38 / 47 |
| steering MAE-to-target at point 9, N = 1 / 5 / 10 | 78.4 / 8.7 / 3.1° → 75.3 / 6.7 / 3.3° | 0.69 / 0.16 / 0.08 → 0.51 / 0.08 / 0.08 m/s | 1.77 / 0.40 / 0.23 → 1.60 / 0.25 / 0.20 m/s² |

The paper-protocol K for direction drops from 46 to 35 because that count reads the test split at every round and
the test split is larger; the nested K barely moves. Part 2's contiguous design was also rerun at 70/30
(`results/split70_p2/`). The direction radius gap is +0.26 / +0.24 at points 12 / 22 (80/20) and +0.25 / +0.26
(70/30). The point-22 endpoint loss of the spline (+3.80°) shrinks to +0.47° at 70/30, consistent with the arc sweep
(§4.3), where the loss appears on 3 of 16 arcs. For speed and acceleration the bend is 0.04–0.19× centroid noise at both
splits, so the verdict flickers on sub-margin differences, and "scalars are straight, no independent-readout gain"
holds at both. The split of record stays 80/20 so that Part 1 and Part 2 read the same clips.

## 4. Part 2: spline steering

### 4.1 The circular structure

Recipe (Goodfire A.3): PCA-64 on train, one centroid per value, a periodic cubic spline for direction and a natural
spline for scalars. The intrinsic angle is atan2(PC2, PC1) on the centroids, computed without labels and then checked
against θ.

| Direction layer | unsupervised angle vs θ: circ. corr (mean / max dev) | supervised circular chart MAE, radius | centroid PR / residual PR | expected sagitta ÷ centroid noise at 45° / 90° gap | LOO cubic beats line |
|---|---|---|---|---|---|
| 8 | failed in both planes → labels (flagged) | 7.1°, 3.17 | 3.75 / 7.99 | 0.15 / 0.59 | at 90° only |
| 12 | −0.983 (7.5° / 15.1°) | 3.5°, 5.56 | 3.41 / 7.84 | 0.22 / 0.85 | at 90° only |
| 22 | −0.964 (10.9° / 28.7°); pipeline plane (clips): 0.948 (11.8° / 34.1°) | 5.8°, 7.27 | 3.53 / 7.58 | 0.19 / 0.74 | never |

Source: `p2_geometry_direction_L{8,12,22}.json` (`angle.centroid`, `angle.activation`). The first figure in each cell
is the circular correlation between θ and the label-free angle atan2(PC2, PC1) in the centroids' top-2 PC plane. The
sign of −1 is an orientation flip, which is allowed.

- **The ring is found** (`figures/fig4_centroid_plane_direction_L12.png`). The headline is the label-free angle above:
  circular correlation −0.983 with θ at point 12 and −0.964 at point 22. Separately, at point 12 the supervised chart
  and the centroid PC plane agree: the two angle assignments over the 64 centroids have circular correlation 0.994
  (mean deviation 4.7°). That number is agreement between two planes, not a correlation with the labels. The chart's
  principal angles to the top-2 PCs of the clip activations are 5.6° and 73.2°, so only one axis is shared: the ring
  is dominant among centroids, not among clips.
  The interpolating spline through all 64 centroids overshoots (visible loops in the figure; held-out reconstruction
  error on the contiguous arc 188 vs 2.8 for the smoothing spline), so all steering uses the count-weighted smoothing
  spline, chosen on train folds.
- **Curvature vs noise.** Over any knot gap up to 45° the chord and the arc differ by less than a quarter of centroid
  noise. Held-out centroids on the contiguous 45° arc are rebuilt best by the chord at every direction layer (point
  12: chord 2.58, smoothing spline 2.82).
- **Planted-ring positive control** (point 12). A synthetic ring is recovered without labels once its radius is ≥ 0.40
  of the real ring's (circular correlation 0.996). A cubic-over-line gain appears only at 0.80[^planted]. A
  "curvature below noise" result is therefore a real null for this pipeline, not blindness.
- **Cone check / velocity plane.** On the speed set the ring's radius grows with speed and then saturates: at point
  12 it is 4.12 at 0.46 m/s and 7.99 at 3.79 m/s (ratio 1.94 for an 8.29× speed ratio; correlation 0.81), and 7.2–8.1
  from 1.4 m/s upward. Procrustes fits of the (direction × speed) cell centroids favour the ring over the velocity
  plane at every layer, narrowly (point 12: 0.445 vs 0.428; point 22: 0.435 vs 0.421; point 8: 0.285 vs 0.285). The
  sharp test is to take a chord between opposite directions: a velocity plane predicts the speed readout at the
  midpoint collapses (ratio cos 90° = 0), and a ring predicts it is unchanged. Measured MLP-speed ratios at Δθ = 180°
  are 0.989 / 0.993 / 1.071 at points 8 / 12 / 22, and Eq. 9 speed ratios 0.991 / 0.963 / 0.953. **Verdict: ring,
  with a radius that saturates in speed, not a velocity plane**[^vp] (`figures/fig4_ring_radius_vs_speed.png`).
- **Two routes to θ + 180°** (exploratory; the spline is built on all 64 values). Walking the spline from θ to θ + 180°
  one way or the other, the midpoint reads +89° (MLP; probe +90°) on the +90° route and −94° (probe −90°) on the other,
  with 98% of carriers on the predicted side at point 12. At point 22 the numbers are +86° / −83° (97% / 95%). The
  chord's midpoint collapses (probe radius 0.14 vs 0.99 at the source at point 12) and reads no consistent angle
  (circular SD 110° probe, 140° MLP). The endpoints are identical by construction (MLP error 26° at point 12)[^tr]
  (`figures/fig4e_two_route_direction_L12_L22.png`).
- **The speed axis does not rotate with direction (cylinder).** A speed probe fit within 22.5° direction bins predicts
  held-out speed worse than one global axis (R² 0.863 vs 0.979 at point 12; 0.921 vs 0.983 at point 22). The local
  axes do tilt more than same-size random bins (|cos| to global 0.69 vs 0.91; 0.56 vs 0.90), so there is a shared axis
  plus a smaller direction-dependent part. A planted rotating code is flagged by the same detector (local R² 0.859 vs
  global −0.066)[^rot] (`figures/fig4f_rotating_speed_axis_L12_L22.png`).
- **Speed and acceleration are straight.** Knot spacing along the speed curve is linear (R² 0.999, vs 0.909 for
  log). Centroid PR is 1.65 / 1.67 for speed (points 12 / 19) and 1.61 / 1.66 for acceleration (12 / 21). The line
  beats the cubic at every stride ≤ 8. This negative was pre-registered.

### 4.2 What a meaningful held-out evaluation is

Three leaks and the separation used for each:

1. **The spline memorises the knots.** Held-out label values are never used as knots. There are three designs:
   *scattered* (every 4th value; largest gap 11.25°, where chord and arc coincide, so this design cannot separate the
   methods), *contiguous* (a 45° arc of 8 directions, 303.75°–343.125°, or an interior block of 8 speeds or 8
   accelerations; the spline-vs-line claim rests on this design), and *extrapolation* (the top 8 speeds or
   accelerations, labelled). Only at a held-out value can an endpoint readout separate the arms. There the line aims
   at a chord point and the spline at a curve point, and they differ by the sagitta: 0.13–0.45 of centroid noise on
   the direction arc at point 12[^steer].
2. **The readout is the intervention.** The data are split three ways: knot clips (folds 0–2, 632 clips) build the
   spline, probe clips (folds 3–4, 480 clips) fit the evaluation probe, the MLP and the Eq. 9 reference, and test
   clips are steered (48 per target, 384 steers). Beside the probe there are two readouts that did not build the edit:
   agreement R with real-clip centroids at the target value (`nearest_real_R`), and an MLP on disjoint clips (§4.4).
3. **The edit looks right only at layer L.** This leak needs the propagation and predictor readouts (§4.5).
   At the steered layer, Goodfire's Eq. 9 "behaviour" is a softmax over distances to the spline, so it restates the
   activation geometry. Every figure that uses it prints that caveat.

"Held-out" is used in three senses, and each result says which it meets. *Excluded from fitting* holds for all
Part 2 results. *Excluded from development decisions*: the spline type, k, the number of waypoints and the layers were
chosen on train folds. *Untouched*: test is read once for the table below, and anything chosen after that read is
labelled exploratory. Points 12 and 22 were chosen on train geometry; point 12 is labelled "exploratory" in the JSON
because it is neither the onset, the paper's layer nor the peak.

Design of the arms. Both arms edit the same PCA-64 subspace and add back each clip's identical off-subspace residual
(matched support). Goodfire's own linear baseline erases the residual, so it is run separately and labelled. The
controls are a dose-matched line (rescaled to the spline's ‖Δ‖ at each waypoint), a *projected* arm (the chord walked
with the spline's spacing), a *reflected* arm (the bend flipped), 20 endpoint-matched random curves, 20 shuffled-centroid
curves, and a BF16 repeat. Each path has K = 50 waypoints.

**The verdict rule is post hoc.** The rule that turns gaps into a call (`verdict.call` in each steering JSON) was
iterated during development, after looking at results. An early energy-based comparison was dropped as a deciding
metric once it was seen to reward residual erasure (the replace arms are on-curve by construction, so Eq. 9 favours
them). The final rule is symmetric: a gain or a loss counts only beyond the same practical margin, and an endpoint
loss beyond that margin overrides path gains ("negative_endpoint"). It was frozen at commit 8d3cac8, before the
multi-arc sweep and the 70/30 reruns. The frozen rule was applied unchanged to all 32 arc runs and to every steering
file cited here. The calls should still be read as a summary of the gaps and CIs, which are reported directly.

### 4.3 Results

**Direction, contiguous 45° arc** (manifold = spline, linear = chord; gaps are manifold − linear with a 95% paired
clip bootstrap):

| | pt 12 spline | pt 12 line | gap [CI] | pt 22 spline | pt 22 line | gap [CI] |
|---|---:|---:|---|---:|---:|---|
| endpoint probe error | 9.73° | 9.62° | +0.11 [0.05, 0.18] | 10.69° | 6.89° | +3.80 [3.36, 4.23] |
| nearest-real agreement R | 0.196 | 0.197 | −0.0004 [−0.0018, 0.0008] | 0.175 | 0.180 | −0.005 [−0.006, −0.003] |
| min readout radius along path | 0.86 | 0.61 | +0.26 [0.22, 0.29] | 0.85 | 0.61 | +0.24 [0.21, 0.27] |
| Eq. 9 energy ÷ real-clip floor | 0.84 | 1.42 | | 1.01 | 1.00 | |
| intermediate mass on the arc | 0.68 | 0.48 | | 0.66 | 0.45 | |
| waypoint ordering (Spearman) | 0.90 | 0.79 | | 0.76 | 0.67 | |
| reflected arm: radius / energy / ordering | 0.54 / 1.64 / 0.52 | | | 0.51 / 1.12 / 0.50 | | |

Source: `p2_steer_direction_direction_L{12,22}_contiguous.json`. Figures:
`figures/fig4_waypoint_readout_direction_direction_L12_contiguous.png` (radius and Eq. 9 distance along the path) and
`figures/fig4_path_energy_direction_direction_L12_contiguous.png`.

**Sixteen held-out arcs** (the contiguous design repeated with seeds 1–16, each holding out a different 45° arc of 8
directions; points 12 and 22; `results/arcs/L{12,22}_s{1..16}/`; `figures/arcs/`). Seeds 4 and 8 drew the same arc
(258.75°–298.125°), so there are 15 distinct arcs in 16 runs. Mean ± SD across the 16 runs of the paired gap
(spline − line, `gaps.manifold_minus_linear`):

| | pt 12 | pt 22 |
|---|---|---|
| min readout radius gap | +0.26 ± 0.05 (range +0.13 to +0.31) | +0.24 ± 0.07 (+0.11 to +0.32) |
| endpoint probe error gap | +0.08° ± 1.47 | +1.26° ± 3.54 |
| endpoint gap, duplicate arc counted once | −0.09° ± 1.35 | +0.68° ± 2.77 |
| verdict "negative_endpoint" | 0 / 16 | 3 / 16 |

The path result holds on every arc at both points. The endpoint ties at point 12. At point 22 the spline is slightly
worse on average; the three "negative_endpoint" arcs are the duplicated 258.75°–298.125° arc (+9.3° and +9.9°) and
191.25°–230.625° (+3.7°), while the other 13 runs are within ±3°. The single-arc point-22 loss in the table above
(+3.80°) is one of the larger values in that spread, not the typical case.

**Donor ceiling** (point 12 / 22, same carriers). Replacing the carrier's PCA-64 coordinates with those of a real
unseen test clip at θ\* reaches nearest-real R = 0.23 / 0.20, against 0.19 / 0.17 for both spline and line. Swapping
the whole activation adds almost nothing to R (+0.004 / +0.001). A real clip at θ\* reaches only R ≈ 0.2 against the
other real clips at θ\*, so this readout's ceiling is set by clip-specific variance, and the spline reaches 82% / 84%
of it. The MLP evaluator reads the in-subspace donor at 25.2° / 18.5° vs 30.7° / 21.2° for the spline[^donor]
(`figures/fig4d_donor_ceiling_direction_L12_L22.png`).

**All designs, endpoint probe error, spline vs line** (source `p2_steer_{var}_{var}_L{pt}_{design}.json`):

| Variable, point | scattered | contiguous | extrapolation | held-out context |
|---|---|---|---|---|
| direction 12 | 7.00° vs 7.05° | 9.73° vs 9.62° | — | speed-set clips: 10.89° vs 10.92° |
| direction 22 | 5.86° vs 5.86° | 10.69° vs 6.89° | — | 11.08° vs 7.57° |
| speed 12 (m/s) | 0.156 vs 0.153 | 0.163 vs 0.133 | 3.22 vs 0.158 | |
| speed 19 | 0.083 vs 0.082 | 0.092 vs 0.080 | 0.49 vs 0.23 | |
| acceleration 12 (m/s²) | 0.345 vs 0.345 | 0.309 vs 0.305 | 1.66 vs 0.67 | |
| acceleration 21 | 0.279 vs 0.277 | 0.282 vs 0.266 | 0.87 vs 0.74 | |

Position sheet (speed set, start (x, y), thin-plate spline vs chord to a held-out interior 2×2 block): at point 12
"negative: TPS path indistinguishable from chord". At point 19 "chord better than the TPS path on err_path and
excess_to_nearest_real". Endpoint error is 0.178 m for both vs 0.197 for a Delaunay interpolation (point 12)[^sheet].

**Reading.** At the encoder layer, the spline stays on the ring (the readout radius stays near 0.86 and the Eq. 9
distribution walks the arc in order) and the line cuts through it (the radius drops to 0.61). The reflected arm is
worst on every path metric, so the bend helps in the right direction and not merely by being a bend. That is the
Goodfire picture reproduced at held-out values. But readouts that did not build the edit do not separate the arms.
Nearest-real agreement ties at point 12 and slightly favours the line at point 22. The endpoint probe error ties at
point 12. At point 22 the line is 3.8° better on this arc, but across 16 arcs the loss appears on 3 and averages
+1.26°. The Eq. 9 energy advantage at point 12 is gone at point 22 (1.01 vs 1.00). For speed and acceleration, spline,
projected and reflected arms coincide with the chord. Where they differ (extrapolation) the spline is worse, because a
smoothing spline extrapolates badly. Later layers and the predictor are tested in §4.5.

### 4.4 Controls and the comparison with Part 1

- **Random curves** (20 endpoint-matched draws, point 12 contiguous). Endpoint readouts match by construction. On the
  path, the spline ranks 1/21 on off-curve excess (0.15 vs a band of 0.68–0.88) and on Eq. 9 energy (0.92 vs
  1.48–1.75). Unmatched random curves have endpoint error 87.9°. **BF16**: the winner on both energy metrics is
  unchanged. **Dose-matched line**: endpoint 9.41°, energy 1.27, radius 0.61, so the line's deficit is not a matter
  of dose.
- **Goodfire's own linear baseline** (the whole activation replaced by a chord point). Nearest-real R is 0.625 and
  endpoint error 1.52°, against 0.547 for Goodfire's manifold arm and 0.196 for our additive arms. Erasing the residual
  makes the activation look much more like the target centroid, so Goodfire's comparison mixes "residual erased" with
  "curved vs straight".
- **Bake-off at matched edit norm** (all arms rescaled per clip to the spline's ‖Δ‖; errors from the linear probe /
  an MLP on disjoint probe clips; unsteered 88.9° / 92.3°)[^bake]:

| Arm (nominal rank) | pt 12 probe / MLP | pt 22 probe / MLP |
|---|---|---|
| spline (64; effective 2.55) | 9.7° / 30.9° | 10.7° / 21.3° |
| chord (64) | 9.4° / 32.6° | 7.5° / 19.7° |
| centroid transport x + μ(θ\*) − μ(θ) | 6.4° / 32.0° | 8.1° / 20.4° |
| ring rotation (2) | 16.9° / 36.7° | 35.5° / 39.3° |
| Part 1 probe-QR least squares (34 / 84) | 5.6° / 23.1° | 12.6° / 19.3° |
| nearest-centroid snap | 21.9° / 29.1° | 21.5° / 21.5° |

At matched norm, Part 1's multi-probe subspace edit moves the linear readout furthest at point 12. Every arm leaves
an independent MLP evaluator at 19–39°, even where the linear probe reads 5–10°. A rank-2 ring rotation does not
match the higher-rank edits. **Strengths of the spline**: it closes around the circle, finds its coordinate without
labels, and keeps intermediate states on-distribution by every geometric measure. **Limitations**: at held-out
endpoints it offers nothing a chord does not, it overshoots through the full knot set (hence smoothing), and it
extrapolates badly on scalars. **Failure cases**: the point-22 endpoint on 3 of 16 arcs (the line is better by
3.7–9.9°), scalar extrapolation, and the position sheet.

- **Cosine between the Part 1 step and the spline** (additional metric; contiguous design, same rows; the Part 1
  step is the multi-probe subspace edit x\* − x at that point)[^cos]:

| Variable, point | Part 1 step · spline net displacement | spline tangent · chord: start / midpoint / end |
|---|---:|---|
| direction 12 / 22 | 0.66 / 0.54 | 0.31 / 0.81 / 0.41 and 0.30 / 0.76 / 0.37 |
| speed 12 / 19 | 0.51 / 0.45 | 0.66 / 0.82 / 0.50 and 0.76 / 0.91 / 0.73 |
| acceleration 12 / 21 | 0.53 / 0.61 | 0.82 / 0.96 / 0.73 and 0.77 / 0.90 / 0.75 |

  The Part 1 step points roughly along the spline's net displacement (cosine 0.45–0.66), so the two methods move the
  activation in related but not identical directions. For direction the spline tangent bends away from the chord at
  both ends (0.31 and 0.41 at point 12), which is the ring's curvature seen in full activation space. For the scalars
  the tangent-chord cosine stays at 0.73–0.96, except on the speed path at point 12 (0.66 at the start, 0.50 at
  the end) (`figures/fig_cosine_tangent_{direction,speed,acceleration}.png`).
- **Shared vs clip-specific edit.** The fraction of edit energy shared by all steered clips (n‖mean Δ‖² / Σ‖Δᵢ‖²,
  `shared_delta_fraction`) is 0.41 for the spline and 0.37 for the line at point 12, and 0.44 / 0.39 at point 22.
  An earlier project found edits that were ~99% a constant shift, and `PART2_RATIONALE.md` planned for that case.
  Here more than half of every edit is clip-specific, so spline and line are not merely one constant shift plus a
  small correction.

### 4.5 Beyond the steered layer (GPU session 2)

Design[^s2]: 200 test carriers × 4 targets on the held-out arc × 6 arms (probe-QR, radius-matched, interpolating
spline, smoothed spline, chord, random at matched ‖Δ‖) at points 2 / 8 / 12 / 22, plus shuffled-target controls. Each
carrier has a pixel twin rendered at the target direction (renderer validated on 20 supplied clips: disk IoU mean
0.985, min 0.975). Zero edits reproduce the unedited forward pass to ≤ 6e-6 relative error at every point.

**Propagation** (a direction probe refit at every later point on probe clips; error to target in degrees, unsteered
≈ 91°; source `session2_propagation.json`, `readout_a`):

| Steer point | read at the steered point: probe-QR / smoothed spline / chord / random | read at point 25: same arms | point 25, scored against the shuffled neighbouring target |
|---|---|---|---|
| 2 | 11.7 / 15.3 / 15.6 / 89.4 | 90.9 / 90.9 / 90.9 / 90.8 | 90.9 / 90.9 / 90.9 / 90.8 |
| 8 | 6.4 / 8.0 / 6.5 / 78.5 | 90.1 / 90.4 / 90.4 / 90.6 | 90.1 / 90.5 / 90.4 / 90.6 |
| 12 | 4.1 / 14.3 / 5.1 / 91.2 | 86.5 / 86.8 / 87.3 / 90.6 | 86.8 / 87.2 / 87.6 / 90.5 |
| 22 | 3.6 / 10.7 / 3.6 / 89.8 | 17.4 / 25.2 / 21.8 / 100.1 | 23.6 / 26.6 / 23.1 / 99.8 |

Edits at points 2–12 wash out: four blocks after the edit the error is already 82–88° for every edit arm except the
overshooting spline (for example point 12 → 16), and at point 25 MAE-to-true is back to 4–7°. At the steered point the
arms are read by a probe of the same family that built them, so those columns are not independent evidence. Point-22
edits survive the last three blocks (15–25° at point 25), with weak target specificity. Scored against the
neighbouring target instead of the true one, the error rises by only 1–6° at point 25, against 7–13° at the steered
point. All targets lie in the held-out 45° arc (neighbour at most 39° away, 16.5° on average[^s2]), so this control
shows specificity shrinking, not gone. The activation readout says the same: the projection of the edit onto the
twin's real activation change at point 25 is 0.16 / 0.23 / 0.22 (probe-QR / smoothed spline / chord) against 0.15 /
0.21 / 0.21 for the shuffled twin (`readout_b`). The interpolating spline overshoots (‖Δ‖ = 7.2× the natural twin
change at point 12, vs 0.5–0.6× for the other arms) and is excluded from the reading.

**Predictor** (context frames 1–8 edited at every token, predictor forecasts tubelets 4–7; R = projection of the
forecast change on the real twin-minus-source future change; source `session2_predictor.json`). **This test is
uninformative, not negative.** Probes fit on the encoder's real future tokens (direction 3.4° on them) read the
unedited forecast 61° from its source clip's direction and 68 px from its position[^fpos]. The *real rendered twin's*
context recovers only R = 0.0185 [0.0173, 0.0195] of the real future change (0.073 on disk tokens), so even a perfect
edit would register at R ≈ 0.02. The values:

| Arm | R, point 12 (shuffled target) | R, point 22 (shuffled target) |
|---|---|---|
| probe-QR | 0.00024 (0.00023) | 0.00178 (0.00173) |
| smoothed spline | 0.00049 (0.00043) | 0.00241 (0.00220) |
| chord | 0.00040 (0.00036) | 0.00243 (0.00225) |
| random, matched ‖Δ‖ | 0.00001 (0.00002) | 0.00056 (0.00055) |

At points 2 and 8 every arm is ≤ 0.00011. No arm beats its own shuffled-target control: every pair of 95% CIs
overlaps, and the point-12 interpolating spline, the largest value (0.0057), equals its control (0.0057). The learned
arms do beat the matched random edit at point 22. None of this says the arms are equivalent downstream. The test
becomes informative with a readout fit on predictor outputs and a shuffled null 180° away.

**Time-reversed clips** (the forward-trained probe read on reversed clips)[^trev]. From point 1 on, the direction probe
reads θ + 180° on the reversed clip: the error to θ + 180° is 20.5° at point 1 and 5.7–10.3° from point 2 on, with
98–100% of clips closer to the flipped angle. The probe tracks motion direction, not position or occupancy (a reversed
constant-velocity clip has the same frame set). A speed probe transfers to reversed clips (R² 0.964–0.981 vs
0.936–0.987 forward at points ≥ 1).

**Future-position audit** (disk-centroid probe on the encoder's future tokens: 6.7–8.4 px test error)[^fpos]. Read on
the predictor's forecast, the same probe is 67.9 px from the source's true future position, so the forecast tokens do
not carry position in the encoder's code. Edits move the decoded forecast position by ≤ 0.7 px at points 2–12 (16 px for
the overshooting spline) and by 2.9–5.5 px at point 22, against a 58 px source-to-twin distance. The error to the
twin's position equals the error to the shuffled twin's within 0.4 px for every arm.

## 5. Beyond the three variables

| Question | Result | Source |
|---|---|---|
| Object permanence | Direction decoded from time steps whose frames contain no disk (89 clips; test 15 clips / 22 tokens): MAE 7.5° [5.6, 9.4] at point 8 (visible 5.4°), 6.1° at point 22 (visible 3.8°); shuffled-label null 84.5°, p = 0.001. The random-init encoder, same clips and protocol, does as well: test-clip absent-step MAE 5.9–7.2° across points vs 6.1–13.4° for V-JEPA 2 (null ≈ 90° for both). V-JEPA 2 is ahead only late, by ≤ 1.0° on test clips (points 16–25) and 2.4° / 2.0° pooled at points 22 / 25; it is behind at points 1–12. On visible steps V-JEPA 2 is 5–7° better from point 8 on. So above-null decoding after the disk leaves is attention mixing within the clip (no causal mask), not learned carrying. | `p1a_object_permanence.json` (`random_init.side_by_side`), `fig6_object_permanence.png` (random-init overlaid) |
| Cartesian vs polar | On constant-velocity clips (596), (vx, vy) reaches onset at point 1 and (sin θ, cos θ) at point 2 (difference −1, CI [−1, −1]); block 1 R² 0.929 vs 0.863. Speed set: 0.985 vs 0.855. The one-block "emergence" of direction is the normalisation v/‖v‖. | `p1a_support_onset_*_meanpool.json`, `fig1d` |
| Direction transfer (held-out context) | Direction probe fit on the direction set, read on the speed set at point 9: MAE 4.4° (source CV 4.0°); 8.7° below 1 m/s, 3.3° at 1–4 m/s. On the acceleration set: 5.8°. At point 1: 10.8° (23.9° below 1 m/s). | `p1a_support_transfer_meanpool.json`, `fig1c` |
| Spatial generalisation | Train on start x < 0, test on x > 0, point 9: R² 0.971 (MAE 4.9°), vs 0.972 within-side. At point 22, mean-pool 0.957 vs disk-pool 0.988. | `p1a_support_spatial_{meanpool,diskpool}.json`, `fig1e` |
| Direction vs speed subspace (paper C.4 method) | Overlap direction←speed 0.0740 at point 8 (random expectation 0.0781, 5–95% band 0.0756–0.0808); 0.0733 at point 9 (0.0723, band 0.0694–0.0740). Direction vs acceleration 0.0762 and 0.0739, inside or at the edge of the band. The INLP bases are as orthogonal as random ones, yet steering direction still moves the speed readout (§3.3 off-target). | `step2_subspace_angles.json` |
| Objective axis | V-JEPA vs random-init at the direction peak: probes needed to reach ≤ 10° MAE 4 vs 10; nested K 88 vs 26. VideoMAE matches V-JEPA 2 on all three variables with the same onsets (§3.1): 4 probes to the bar, nested K 67, peak 0.992. | `objective_axis.json`, `fig5_objective_axis.png` |
| Position sheet | Start (x, y) is decodable; 36-cell centroid PR 7.46 (point 12) / 3.76 (point 19), Procrustes to (x, y) 0.38 / 0.66; spline steering gives no path advantage (§4.3). | `p2_sheet_speed_L{12,19}.json` |

## 6. What this says about her framing

- **Detection vs use.** Detection is easy on this data. A random network, and random features of a 32-number
  trajectory, detect all three variables at R² ≥ 0.85. Use is where the evidence thins. The Part 1 edit moves a
  held-out linear probe, but it beats a random basis only past N ≈ 5, works in an untrained network too, and leaves an
  MLP on disjoint clips 19–31° off. On the ladder in `PART2_RATIONALE.md` §2, this project reaches rung 3 at the
  steered layer and rung 4 only by circular measures. Rung 5 was tested: edits at points ≤ 12 wash out, point-22 edits
  survive with weak target specificity, and the predictor readout had no power to see an edit (§4.5).
- **Internal world model vs stimulus-response.** For this stimulus class the random-init control gives a direct
  answer to her question. Linear availability of direction, speed and acceleration is architectural (one attention
  block over rotary positions plus a nonlinearity). What V-JEPA's training adds is precision (direction 10.7° → 3.0°
  vs flat for the random network) and a ring that is recovered without labels at mid depth. Carrying direction into
  disk-free tokens is not a training effect: the random-init encoder does it as well. VideoMAE matches V-JEPA 2 on
  every Part 1 measure, so none of this is specific to latent prediction, and none of it shows that the variables are
  used to predict. We have two points on the training axis (random init, final checkpoint), not a trajectory. The
  authors' OpenReview response states that "all 13 models encode motion direction (R²≥.43), regardless of objective",
  classification CNNs included, so availability is their own finding; training buys precision, fewer probes to steer
  (4 vs 10 to reach 10°[^obj]) and a label-free ring.
- **The linear representation hypothesis: right about the subspace, wrong about the moves.** Direction is linearly
  decodable through (sin, cos), which is a 2-D linear subspace with a ring on it, and nothing in Part 1 contradicts
  the subspace form of the hypothesis. The steering corollary is what fails geometrically: the straight path between
  distant directions passes through the ring's interior (readout radius 0.61) where the curved path does not (0.86).
  The measured caveat is that the model's own independent readouts do not care, at the steered layer or downstream
  (§4.5). The precise statement supported is "for a cyclic variable the hypothesis describes the subspace and
  misdescribes the moves, geometrically". A functional difference was not found; on this stimulus the predictor test
  is too weak to rule one out.

## 7. Limitations and next steps

- **Stimulus.** A single disk on a flat background is nearly pixel-decodable. The main non-reproduction may be a
  property of the data: the hard set moves the direction onset to point 5 (§3.1). That set has 392 clips and only 8
  directions, covers direction only and uses one render seed, so it is a pointer, not a reproduction.
- **Training dynamics.** With intermediate V-JEPA 2 checkpoints, the random-init vs final contrast becomes a curve.
  That is the natural test of when precision and the ring appear.
- **Predictor test uninformative.** Probes fit on real tokens read the unedited predictor forecast 61° off in
  direction and 68 px off in position, and the real twin's context recovers only R = 0.0185, so the test had no power
  to see an edit; its shuffled targets are at most 39° apart. Next: a readout fit on predictor outputs and a 180°
  shuffled null.
- **Post-hoc verdict rule.** The spline-vs-line call was iterated after seeing results and frozen at 8d3cac8 before
  the arc sweep and the 70/30 reruns (§4.2). The gaps and CIs are the evidence; the calls summarise them.
- **Sample size vs d.** Around 1,200 train clips against d = 1,024 makes K a ridge count at a CV-chosen α. The K
  values should be compared across layers only at a fixed α (see the caveat in `p1b_*_dims.json`).

## 8. Reproducibility

- **Commands.** `scripts/extract.py` (GPU; activations to `artifacts/activations/`), `scripts/make_splits.py`,
  `scripts/run_step1.py [--dataset --variable --pool --model --shuffled]`, `run_pixel_baseline.py`,
  `run_random_feature_floor.py`, `run_step1_paperscale.py`, `run_step1_support.py`, `run_step2.py`, `run_step2_dims.py`,
  `run_step2_angles.py`, `run_step3.py`, `check_probe_recipe.py`, `run_geometry_checks.py`,
  `run_part2.py --dataset --layer --holdout {scattered,contiguous,extrapolation} --spline smooth`, `run_bakeoff.py`,
  `run_velocity_plane.py`, `run_position_sheet.py`, `run_object_permanence.py`, `run_objective_axis.py`,
  `run_donor_ceiling.py`, `run_two_route.py`, `run_rotating_speed_axis.py`, `run_cosine_tangent.py`,
  `render_hard_stimuli.py`, `make_figures.py`. Session 2: `scripts/session2_box.sh` → `run_session2.py`,
  `session2_extras.py`. The 70/30 reruns set `WM_SPLIT_PATH=splits/split_paper70.json`.
- **Provenance.** Every results JSON records the split sha256 (`98e6310c…`), seeds (split 0, all others 0), git
  commit and a dirty flag. The 20 `p2_steer_*` files and all 32 arc runs were produced at commit 8d3cac8; the other
  Part 2 files at 677b305, 8e552c1, 8829195, 8f08444 or fbf4f72; all with `git_dirty_src_or_scripts: false`. Most
  Part 2 files record split and source paths inside the frozen scratchpad worktree that ran them; the split sha256 and
  the commit are the same as the repository's. The session 2 files were scored at b9c53d0 or 0f34ec2 with the dirty
  flag set.
- **Numerics.** CPU–GPU parity on 8 clips: worst per-layer max|Δ|/max|x| 8.2e-5 (rule < 1e-3); GPU batch-8 vs
  batch-16 gap 1.31× the CPU–CPU gap (rule ≤ 2×). Frame hashes and disk masks match, and 27/27 sha256 checks of the
  downloaded activations pass[^gpu].
- **Cost.** GPU session 1 (RTX 4080 SUPER, Vast): 37.7 billed minutes, $0.19 of compute[^gpu]. Session 2 box (RTX
  4060 Ti, $0.198/h): $0.512 from start to the end of session 2 (the session 2 run itself $0.311), and $0.637 billed as
  of 19:57 ET with the box still running follow-on VideoMAE and random-init extraction, plus $0.006 egress[^cost]. At
  that rate the session 2 box comes to about $0.65–0.70 when stopped. No other box's cost is recorded in `results/` or
  `artifacts/`.
- **Tests.** `pytest --collect-only` collects 173 tests at the commit of this report.

[^gpu]: `artifacts/gpu_session1.json`.
[^ptxt]: Line numbers in `refs/physics_paper.txt` (text of arXiv 2602.07050): (a) 1211 vs 1172; (b) 1207 vs 1243 and 1266; (c) 1245–1246; (d) 430–431 vs 1256 and 1269. Our numbers: `p1b_*` (`K`, `K_loose`), `results/split70/COMPARISON.md`, `p1c_direction_L9_strict.json` (`strict_eval`), `p1c_direction_L9.json`.
[^s1]: `results/p1a_direction_direction_meanpool.json` (n_train 1200, n_test 300); `results/p1a_paperscale_speed_speed.json` (speed train 1228).
[^steer]: `results/p2_steer_*` (n_test_clips 308 for speed/acceleration; `sagitta_per_target`; `n_knot_clips` 632, `n_probe_clips` 480).
[^recipe]: `results/p1a_probe_recipe_check.json`.
[^disk]: `results/p1a_direction_direction_diskpool.json`.
[^obj]: `results/objective_axis.json` (`selectivity_at_peak`, VideoMAE rows); `results/p1a_*_meanpool_videomae.json`, `p1b_*_videomae_L*.json`, `p1c_*_videomae.json`.
[^ps]: `results/p1a_paperscale_direction_direction.json`.
[^pss]: `results/p1a_paperscale_speed_speed.json`.
[^adam]: `results/p1b_{direction_direction,speed_speed}_meanpool_L{8,9}_adam_{b64,full}.json` (`K_first`, `sawtooth.by_metric`, `rounds[].failed_to_train`).
[^p1cr]: `results/p1c_direction_L9_random.json`.
[^dim4]: `results/p1b_dims_four_ways.json` (`variables.{direction,speed}.layers[].{literal,whitened,leace,dft}`, `control.{clean_ring,sheared_ring_x30,ring_with_k3_harmonic,three_copies}`; DFT column = `dft.frac_k_ge_1`, share of non-constant power); per-layer K and α in `p1b_*_meanpool_dims.json`.
[^strict]: `results/p1c_direction_L9_strict.json` (`strict_eval`).
[^planted]: `results/p2_planted_ring_direction_L12.json`.
[^vp]: `results/p2_velocity_plane.json`.
[^sheet]: `results/p2_sheet_speed_L{12,19}.json` (`verdict`, `summary`).
[^bake]: `results/p2_bakeoff_direction_direction_L{12,22}_contiguous.json`.
[^s2]: `results/session2_plan.json` (`targets`: 4 per carrier, all in 303.75°–343.125°), `results/session2_renderer_validation.json`, `results/session2_stimuli_validation.json`.
[^fig22]: `loose_threshold` in each `p1b_*` file: R² < 0.3 for direction and R² < 0.1 for speed and acceleration, the thresholds of the paper's Fig. 22. For speed and acceleration the K at that threshold equals the nested K at every V-JEPA point except acceleration at onset (493 vs 466). `K_loose_censored` is true only for VideoMAE acceleration at point 22, where K_loose = 87 is a floor (the sequence stopped while R² was still above 0.1); the V-JEPA files predate the flag.
[^stim]: `results/stimuli/{paper_layout,hard}/p1a_direction_direction_meanpool{,_random}.json`; `results/session2_stimuli_validation.json`.
[^s70]: `results/split70/comparison.json`, `results/split70/COMPARISON.md`, `results/split70_p2/COMPARISON.md` and `results/split70_p2/p2_steer_*_contiguous.json`; 80/20 null p from `results/p1c_{direction,speed,acceleration}_L9.json`.
[^tr]: `results/p2_two_route_direction_L12_L22.json`.
[^rot]: `results/p2_rotating_speed_axis_L12_L22.json`.
[^donor]: `results/p2_donor_ceiling_direction_L12_L22_contiguous.json`.
[^cos]: `results/p2_cosine_tangent_{direction_L12,direction_L22,speed_L12,speed_L19,acceleration_L12,acceleration_L21}_contiguous.json` (`cosines.step_net`, `cosines.tangent_chord`).
[^trev]: `results/session2_timerev.json`, `results/session2_timerev_speed.json`.
[^fpos]: `results/session2_future_position.json`.
[^cost]: `results/session2_cost.json` (`cost_start_to_session2_done_usd`, `session2_run_cost_usd`, `cost_so_far_usd`, `pull_egress_usd`, `status: running`).
