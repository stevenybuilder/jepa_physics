# V-JEPA 2 physics take-home: report

Every number below comes from a file in `results/` (including `results/split70`, `results/split70_p2`,
`results/arcs`, `results/stimuli`) or `artifacts/gpu_session1.json`, named in a footnote or in the table's source line.
Paper claims are stated qualitatively and attributed to the paper.

## 1. Summary

The README asks for a small-scale reproduction of Joseph et al. (arXiv 2602.07050): layer-wise probes, iterative
nullspace probing, and multi-probe subspace steering for direction, speed and acceleration in the frozen V-JEPA 2
ViT-L/16 encoder. It also asks for an open-ended Goodfire spline-steering extension (arXiv 2605.05115) that takes care
over the circular structure of direction and over what counts as a held-out steering evaluation. **Step 1 reproduces
in part, and which part depends on the readout.** Mean-pooled probes read all three variables from block 1 with no
late decline, and a random-init ViT-L and random nonlinear features of the disk trajectory do nearly as well. That
holds under every fold grouping and at the paper's clip count, so it is not a sample-size effect. But the paper's own
App. C.5 puts the sharp step in per-patch probes and says mean-pooled probes rise gradually, so the pooled curve was
the wrong comparison. Per patch, three things hold. On the supplied clips the per-patch code is early as well: mean
per-position R² is 0.56 at block 1, 0.89 at block 5 and 0.96 at block 6, with no step at the paper's transition. That
code is training-selective: the random-init network pools to 0.86–0.88 at every depth, but its mean per-position R² never
exceeds 0.39 (best single position 0.76–0.86), the paper's "fragmented local signal that pooling adds up" regime, which V-JEPA 2 leaves within six
blocks. On a harder rendered set (textured floor, shading, smaller disk; 392 clips, 8 directions, three render seeds) the
per-position curve rises most between points 4 and 6 on every seed (+0.24 to +0.29) and only +0.07 ± 0.02 across
points 8→9, where the 90% rule fires on one seed (9) and one point earlier on the other two (8); a random-init network
plateaus at 0.57–0.62 from point 6 on the same clips. What does replicate at the paper's depth on all three seeds is
half-frame transfer: negative at points 7–8 (−1.0 to −1.8 at 8) and positive from point 9 (+1.4 to +1.9 jump), after
0.69–0.73 at point 1 (§3.1). VideoMAE (pixel reconstruction) matches V-JEPA 2 on the pooled curves for all three variables.
**Step 2 reproduces in part.** Every variable needs tens of probes at the paper's layer, far outside a random-removal
band. Measured four ways, that count reflects anisotropy: a whitened erasure needs one probe at every point; the code
is rank-2 linear plus a second harmonic and a nonlinear residual (§3.2). Speed does not need fewer probes than
direction (in raw coordinates and under one common R² stop too, except the paper protocol at point 8), and the ridge curves have no sawtooth. Under the paper's literal Adam
recipe both variables' curves are jagged and K roughly doubles, with no direction-specific sawtooth. **Step 3
reproduces in shape** (one probe fails, a few probes reach the target, MAE-to-true rises), with 3–5 probes to 10°
where the paper needs about 20. How it compares with a random orthonormal basis of fixed rank 2K depends on the
evaluation probe: with a near-unregularised probe (α = 1e-3, or the C.11 Adam recipe; C.12 gives no recipe) the learned basis
beats all 20 draws from N = 5, with my CV-chosen one only from N = 14. Against a rank-matched random basis it first beats all 20 draws at N = 2–7 (by variable
and layer; N = 3 under the near-unregularised probes); at N = 1 it separates from neither. The same curve appears in an untrained
network, and at the onset layer the learned basis never beats the rank-2K random basis for N ≤ 20. Rerunning Parts 1
and 2 at the paper's literal 70/30 split changes no qualitative verdict. **Part 2.** Direction lies on a ring.
My centroid-plane angle recovers it without labels at point 12 but not at point 22; Goodfire's own label-free angle
fails its periodicity test at point 12 and passes at 22, where it sits up to 38–97° from θ; its sequential tasks use
the ordinal index, as A.3 says, and only its 70B cyclic configs take the coordinate from the labels[^src].
The ring is an ellipse, not a circle: axis ratio 0.68–0.89 in its own plane from point 8 on (0.74 / 0.87 / 0.74 at
points 8 / 12 / 22), bent out of that plane by a cos 2θ saddle that holds 20–34% of the centroid variance. Held-out
clips occupy the ring along its whole length and leave it hollow in its plane. At held-out direction values the spline path stays on the ring and the straight path cuts across the
hollow: the minimum readout radius is higher for the spline by +0.26 ± 0.05 at point 12 and +0.28 ± 0.03 at point 22
(mean ± SD over 16 runs covering 15 distinct held-out 45° arcs; point 22 on the labels angle[^src]), and it is higher on
every arc. The endpoint error ties at point 12 (+0.08° ± 1.47) and at point 22 (+0.17° ± 1.33; 1 of 16 runs flagged
"negative_endpoint"). Endpoint readouts that did not build the edit (nearest-real-clip agreement, an MLP on disjoint
clips) do not separate the two; the labels of the real clips nearest the path midpoint do. The spline's advantage over
the chord is in the ring plane and in the forecast, not in distance to real clips in the full subspace, where the chord
is as close at point 12 and closer at point 22. Speed and acceleration are straight, and there the spline adds nothing; in extrapolation it ties the chord once it is
continued along its end tangent, as the authors' code does[^ext].
**Beyond the steered layer (GPU session 2).** Edits at points ≤ 12 wash out within a few blocks and barely reach the
predictor's forecast. Point-22 edits survive to the output with little target specificity. Probes fit on the
predictor's own forecasts show point-22 edits moving the forecast to 11.3–44.2° of the held-out target (the smoothing spline is the 44.2°;
unedited 92.1°[^nat]), position overshooting. At a common edit norm the interpolating spline still beats the chord (19.5° vs 27.2° at
the chord's norm, 12.3° vs 25.7° at the natural one), in forecast angle only. That is a point-22 result. At the encoder
output (the final LayerNorm, the predictor's input and the site Goodfire steers) edits reach the forecast too (chord
12.5°, interpolating spline 15.7°), but at the chord's norm the spline trails the chord by 10.8° and at the natural norm
it leads by only 2.1°[^enc]. Along the point-22 path the forecast follows the
intermediate directions along the spline and jumps along the chord (−13.4° paired, −31.3° at large shifts). The reverse
test does not recover the ring. This is a probe of the forecast on one stimulus and one 45° arc, not a rendered future.

## 2. Setup

- **Model.** `facebook/vjepa2-vitl-fpc64-256` (transformers 4.56.2), frozen. Input is 16 frames at 256², decoded
  with PyAV, divided by 255 and normalised with ImageNet mean/std. There is no resize or crop (`src/wm/extract.py`),
  giving 8×16×16 = 2,048 tokens. Forward passes run in fp32 with TF32 off for matmul and cuDNN (flags recorded as
  false[^gpu]). There are 26 hidden-state points: the embedding, blocks 1–24 (the raw output of block 24, captured by
  a hook), and the final LayerNorm. The paper's "layer L" is our point L+1, so the paper's layer 8 is point 9.
- **Representation.** `meanpool`, the mean over all 2,048 tokens, is used for every Part 1 result, as in the paper.
  `diskpool` (tokens the disk covers) and `timepool` (per time step) are used for controls.
- **Data.** direction 1,500 clips (64 angles, a mix of constant-velocity and accelerating-from-rest clips); speed
  1,536 (64 speeds × 64 directions); acceleration 1,536. The README is explicit: *"The supplied dataset is
  deliberately smaller and simpler than the datasets in the paper. The aim is to reproduce the methodology and
  qualitative findings, not the paper's exact numerical results."* The stimulus is one orange disk on a flat
  background with a fixed camera. Acceleration clips all start at rest, so acceleration, mean speed and displacement
  are one variable here. The paper's acceleration set has the same confound, and I did not try to fix it.
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
| INLP coordinates | no normalisation stated (C.11), raw features implied | train-z-scored features | raw centred coordinates raise nested K 1.4–1.6×; direction vs speed equal at point 9, not at 8 (§3.2) |
| INLP stop rule | R² or MAE rule, whichever fires (C.11) | the same rules | the MAE rule stops speed and acceleration at R² 0.15–0.20, direction runs to R² just under 0.1 (§3.2) |
| Onset | "one-third depth", no numeric rule | first sampled point at ≥ 90% of the maximum, 200-draw clip bootstrap | per-patch curves reported beside the pooled ones (§3.1) |
| CV folds | "5-fold grouped" (App. B), key not stated | stratified by value; direction-, start- and speed-grouped folds rerun | onsets unchanged (§3.1) |
| Per-patch probes | per patch (C.5), method not stated | one probe per spatial position on features averaged over the 8 time steps; a pooled-patch probe; the half-frame test is one pooled probe per half | time structure within a position is not probed |
| Rendered sets | start sampled per (θ, v) pair (App. A) | 7 starts drawn once and shared by every (θ, v) cell | 7 distinct start positions, not up to 392 |
| Steering evaluation probe | fit on test, R² = 0.99 (C.12) | ridge on test, α = 100 by CV inside test (in-sample R² 0.993) | α = 1e-3 or Adam need fewer probes and beat the rank-2K null sooner (§3.3) |
| Objective axis | VideoMAE-v2 family | VideoMAE v1 ViT-L (`MCG-NJU/videomae-large`), 224² | "not the objective" is shown for v1 only |
| Part 2: steering site | Goodfire: last-token residual stream (A.2); encoder output for the world model (§5) | mean-pool over 2,048 tokens at point L | the edited vector is not one the model consumes; §4.1–§4.4 read it with probes, §4.5 adds the edit to every token |
| Part 2: PCA-64 fit set | all prompts in the task (A.3) | knot clips (folds 0–2) at the kept values only | held-out values never shape the subspace; the plane can differ from an all-clip fit (point 22, §4.1) |
| Part 2: spline | interpolating, through the centroids exactly (A.3); √count-weighted smoothing spline for the world model (B.1) | smoothing spline with weight √count / sd_c per knot and coordinate and s = number of knots (both my choices; B.1 gives no smoothing value); interpolating run beside it | interpolating rebuilds held-out centroids worse and its edit is 1.4–1.6× the chord's (§4.1) |
| Part 2: direction coordinate | unsupervised atan2(PC2, PC1) (A.3; the weekdays/months 8B configs inherit `intrinsic_mode: pca`); ordinal index for the sequential tasks (A.3; alphabet/age configs `parameter`); the labels only in the 70B cyclic configs | our centroid-plane atan2 at point 12; labels at points 2, 8 and 22 | label-free only at point 12, and only through our fallback (§4.1) |
| Part 2: manifold arm | replace the PCA-64 part with the curve point (A.6) | additive, x + γ(t) − γ(t_src), residual kept | theirs run as a labelled arm (§4.4) |
| Part 2: base pair of arms | manifold vs whole-activation chord replacement (A.6) | spline vs chord in the same PCA-64 subspace (matched support) | their linear arm erases the residual; run separately and labelled (§4.4) |
| Part 2: waypoints | K = 50 (A.6; the weekdays/months 8B default); alphabet/age 8B configs 150/250 (alphabet_8b_n3 50), 70B configs 100–150, grid/cylinder 20 | K = 50 | E_BC sums over waypoints, so only within-run energy ratios compare |
| Part 2: Eq. 10 temperature | τ = 0.5 on a LayerNorm'd 64-d latent (B.1) | τ = 0.5 in raw PCA-64 units | absolute energies not comparable; τ 0.25–2 keeps the point-12 ordering (`tau_sensitivity` in `p2_steer_direction_direction_L12_contiguous.json`) |
| Part 2: behaviour manifold | smoothing spline through 128 bin centroids (B.1) | interpolating spline through the 64 per-value centroids in the Hellinger tangent plane (A.4), F over 128 bins | circular at the steered layer either way (§4.2) |
| Part 2: carriers | 16 fixed base prompts per task, one set for every pair (A.6) | 48 test clips per target, each steered from its own value | Goodfire starts every carrier at the centroid c_a whatever the carrier's own value (A.6); ours starts each carrier at its true value (an oracle source) and averages over sources |

**Places where the paper contradicts itself**[^ptxt]. (a) The INLP stopping
threshold for direction is R² < 0.1 in C.11 and R² < 0.3 in the Fig. 22 caption: each `p1b` file has both (`K`,
`K_loose`; §3.2). (b) The split is 80/20 in C.11 and 70/30 in C.12: I ran both (§3.4). (c) C.12 fits its evaluation
probe on the test clips it then steers and scores (independent of the steering probes but not of those clips): her
protocol is our §3.3 headline, with a split-half version beside it (12.4° / 17.1° at N = 5 vs 8.7°). (d) The main text
reports < 0.5° with all probe directions, the appendix 11.9° held-out with 20 probes: I report every N. (e) The velocity
set has 392 videos in App. A, but C.12's split is 240 + 103 = 343. (f) The main text measures motion "in pixels per
frame", App. A in m/s. (g) Probe counts disagree across C.10–C.12: C.12 trains "25 probes until R² < 0.1" at layer 8
and reports 20; Table 3 gives a layer-8 direction dimension of 136 (68 probes); C.11 says 14–136 while Table 3 lists 400
at layers 20–23; the main text says 40–50, up to 80. (h) "Speed needs fewer" depends on the layer in the paper's own
Table 3: at layers 0–2 the direction dimension is 30 / 30 / 14 (15 / 15 / 7 probes) against speed's 25 / 24 / 25, so
speed needs more probes there, and fewer from layer 3 on. Goodfire's paper
has a smaller one of its own: A.3 derives the cyclic coordinate as atan2(PC2, PC1) "in an unsupervised manner", and
the weekdays and months 8B configs, the paper's cyclic runs, do inherit that mode (`intrinsic_mode: pca` in
`configs/analysis/activation_manifold.yaml`); the sequential tasks use the ordinal index as A.3 says (alphabet and
age configs `parameter`). The only text-vs-config gap is the 70B weekdays and months configs, which set `parameter`,
the labels, for a model the paper's one-dimensional experiments do not report (A.2: 8B layer 28 "for all tasks").
A.6 says K = 50 waypoints, which the weekdays/months 8B and alphabet_8b_n3 configs use, where alphabet/age 8B use
150/250, the 70B configs 100–150 and the grid/cylinder configs 20.

**Probe-recipe parity** (point = CV-peak layer; pooled out-of-fold R², targets standardised for Adam[^recipe]):
ridge vs Adam (C.11 recipe) is 0.9905 vs 0.9858 for direction, 0.9940 vs 0.9882 for speed and 0.9925 vs 0.9871 for
acceleration. The best coupled-L2 Adam grid cells reach 0.9868, 0.9907 and 0.9887 (AdamW 0.9870, 0.9905, 0.9889).
Every ridge value lies above the Adam CI, so ridge is slightly *better* on all three variables.

## 3. Part 1

### 3.1 Layer-wise probing

**Paper's claim.** Speed and acceleration are decodable early. Direction appears only from about one third of the
depth (the "Physics Emergence Zone"). Performance peaks mid-network and falls toward the output. App. C.5 places the
sharp step in per-patch probes: early direction signal is "fragmented across patches", mean-pooled probes reach
"modest performance" by combining it, and "per-patch probe performance rises abruptly at the emergence zone, while
mean-pooled performance improves more gradually" (`refs/physics_paper.txt` l.981–989).

**Figures:** `figures/fig1_layer_curves.png` (the direction curve jumps at point 1; both motion types are shown in the
right panel); `figures/fig1g_perpatch_direction.png` and `fig1h_perpatch_heatmaps.png` (per-patch probes);
`figures/fig1i_grouped_cv.png` (fold groupings).

| Target | pt 0 | pt 1 | pt 2 | pt 9 | peak (pt) | onset [95% CI] | decline peak→final | CV MAE pt 1 → 9 → peak |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direction (sin, cos) | 0.100 | 0.875 | 0.931 | 0.980 | 0.991 (22) | 2 [2, 2] | 0.0008 | 10.7° → 4.0° → 3.0° |
| speed | −0.002 | 0.983 | 0.983 | 0.988 | 0.994 (19) | 1 [1, 1] | 0.0015 | 0.109 → 0.092 → 0.067 m/s |
| acceleration | −0.019 | 0.977 | 0.975 | 0.982 | 0.992 (21) | 1 [1, 1] | 0.0022 | 0.339 → 0.304 → 0.197 m/s² |
| (vx, vy) | 0.039 | 0.985 | 0.986 | 0.984 | 0.990 (22) | 1 [1, 1] | 0.0007 | |
| (ax, ay) | 0.002 | 0.975 | 0.976 | 0.980 | 0.989 (22) | 1 [1, 1] | 0.0007 | |

Cells are CV R² (5-fold mean). Onset is the first point at ≥ 90% of the maximum, with a 200-draw clip bootstrap for
the CI. Source: `p1a_{var}_meanpool.json`.

**Mean-pooled curve.** Direction is at 0.875 after one block, and no variable declines late. Disk-pooling changes little
(direction peak 0.994, onset still 2[^disk]). The onset does not depend on how the CV folds are grouped (the paper's
App. B says "5-fold grouped" without the key): direction onset is 2 [2, 2] with stratified, direction-grouped and
start-grouped folds, and speed onset 1 [1, 1] with stratified and speed-grouped folds[^gcv]. Per the paper's own C.5,
though, the pooled curve is expected to rise early and gradually, so it is the wrong readout to test the emergence zone
against. The per-patch test is below.

**Controls for the mean-pooled curve.**

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
Availability is there before training; what training buys is precision. The selectivity of V-JEPA over the random network at the direction peak
is 0.125 R²[^obj].

**Paper-scale subsample** (train rows only, 10 seeds where random; `figures/fig1f_paperscale.png`[^ps]):

| Condition (direction) | n clips | V-JEPA block 1 → 8 | onset [CI] | random ViT block 1 → 8 | onset |
|---|---:|---|---|---|---|
| full train | 1,200 | 0.875 → 0.973 | 2 [2, 2] | 0.848 → 0.864 | 1 |
| paper's 8 directions | 150 | 0.590 → 0.933 | 5 [4, 5] | 0.664 → 0.699 | 1 |
| 8 directions, direction-grouped folds | 150 | 0.396 → 0.899 | 5 [5, 8] | 0.647 → 0.695 | 1 [1, 2] |
| 150 random clips, 64 directions | 150 | 0.638 → 0.933 | 5 [3, 6] | 0.715 → 0.737 | 1 |
| 240 random clips | 240 | 0.686 → 0.943 | 4 [3, 5] | 0.734 → 0.750 | 1 |
| 8 directions, constant-velocity clips only | 76 | 0.489 → 0.941 | 6 [5, 8] | 0.694 → 0.746 | 1 [1, 2] |
| same, direction-grouped folds | 76 | 0.361 → 0.818 | 9 [6, 9] | 0.540 → 0.646 | 3 [2, 6] |
| 392 constant-velocity clips, 64 directions | 392 | 0.823 → 0.972 | 2 [2, 2] | 0.847 → 0.863 | 1 |

At 150 clips or fewer V-JEPA's direction onset moves later, to point 4–9, and the rise over the early blocks becomes
selective (the random network stays flat). I first read this as the explanation for the missing emergence zone and
withdraw that reading: at the paper's own clip count the onset comes back to point 2, both for 392 constant-velocity
clips drawn from the supplied set (2 [2, 2] in every seed[^psv]) and for the rendered paper-layout set of 392 clips
(2 [2, 3], below). The late onsets in the small rows come from having few clips per fit, 76–150 against the paper's
392. Speed stays at onset 1 for V-JEPA in every condition. The random network's speed onset becomes unstable
(direction-grouped folds: 0.294 → 0.468, onset 11 [3, 20])[^pss].

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
(0.881 → 0.861). On the pooled curve this is a partial recovery of the emergence zone (point 5 is about a fifth of the
depth); per patch, the hard set shows the paper's half-frame transition at the paper's depth but not its sharp rise
(below). Caveats: 392 clips per seed, 8 directions, direction only, three render seeds, and the 7 start positions are shared by every (θ, v) cell, where the paper samples
starts per pair.

**Per-patch probes (paper App. C.5 / Fig. 18)**[^pp]. One ridge probe per spatial position (16 × 16 = 256), on that
position's tokens averaged over the 8 time steps; a pooled-patch probe fit on all (clip, position) samples and scored
per position; and a half-frame test (one pooled probe fit on the left 8 columns and read on the right, and the reverse).
All scores are test R² on the stored split (rendered sets: 313 / 79).

| Set | per-position mean R², pt 1 / 6 / 8 / 9 / 22 | per-position onset | pooled-patch pt 1, onset | cross-half R², pt 1 / 8 / 9 / 22 | mean-pool onset (test) |
|---|---|---|---|---|---|
| supplied, V-JEPA 2 | 0.56 / 0.957 / 0.958 / 0.975 / 0.95 | 5 [5, 5] | 0.83, 2 [2, 4] | 0.82 / 0.95 / 0.96 / 0.81 | 2 [1, 2] |
| supplied, constant-velocity clips | 0.54 / 0.959 / 0.961 / 0.980 / 0.94 | 6 [5, 6] | 0.88, 2 [1, 2] | 0.87 / 0.96 / 0.97 / 0.84 | 1 [1, 2] |
| supplied, random-init ViT-L | −0.01 / 0.22 / 0.27 / 0.29 / 0.38 | 16 [10, 19] | 0.66, 4 [3, 5] | 0.61 / 0.70 / 0.71 / 0.74 | 1 [1, 1] |
| hard rendered set, seed 0 | 0.49 / 0.806 / 0.851 / 0.944 / 0.96 | 9 [9, 9] | 0.78, 9 [8, 9] | 0.69 / −1.42 / 0.08 / 0.75 | 6 [4, 6] |
| hard, render seed 1 · seed 2 | 0.54 / 0.83 / 0.88 / 0.95 / 0.96 · 0.55 / 0.81 / 0.88 / 0.95 / 0.97 | 8 [8, 9] · 8 [8, 9] | — | 0.73 / −1.03 / 0.39 / 0.74 · 0.71 / −1.84 / 0.08 / 0.79 | — |
| hard, seed 0, random-init ViT-L | 0.41 / 0.57 / 0.60 / 0.61 / 0.62 | — | — | −0.72 / 0.27 / 0.27 / 0.51 | — |
| paper-layout rendered set | 0.47 / 0.958 / 0.955 / 0.978 / 0.93 | 6 [6, 6] | 0.86, 4 [1, 4] | 0.81 / 0.95 / 0.97 / 0.86 | 4 [4, 4] |

Three findings. (1) On the supplied clips both readouts are early. Mean per-position R² is 0.56 at block 1, 0.89 at
block 5 and 0.96 at block 6 (onset 5 [5, 5]; constant-velocity subset 0.54 → 0.87 → 0.96, onset 6 [5, 6]). The
pooled-patch probe works at every position from block 1 (0.83, R² ≥ 0.5 at all 256 positions). Half-frame transfer
is 0.82 at block 1, peaks at 0.96 at point 9 and falls to 0.81 at point 22, the opposite of the paper's late
generalisation. There is no step at the paper's transition (points 8 → 9: 0.958 → 0.975). (2) The per-patch code is
training-selective. The random-init network pools to 0.86–0.88 at every point, but its mean per-position R² never exceeds
0.39 (best single position 0.76–0.86) and its cross-half R² is 0.61–0.74. That is the paper's regime of fragmented local signal that pooling adds up;
V-JEPA 2 leaves it within six blocks on these clips. (3) On the hard set, three render seeds (fresh starts and floor texture, otherwise identical)[^seeds]. The
per-position curve rises most between points 4 and 6 on every seed (+0.286 / +0.245 / +0.237; every position is above
R² 0.5 from point 6 on every seed); across points 8 → 9 it rises +0.093 / +0.065 / +0.065, so the 90%-of-max onset is
9 [9, 9] on seed 0 and 8 [8, 9] on seeds 1 and 2; pooled-patch onset on seed 0 is 9 [8, 9]. A random-init network on
the same clips reaches 0.57 at point 6 and 0.60–0.62 from point 8, so the trained network leads by 0.24–0.35 from point
6 on and not at all at points 1–4, where random matches or beats it (0.41 / 0.54 vs 0.49 / 0.52). Cross-half transfer
is 0.69–0.73 at point 1, negative at points 7 and 8 on every seed (−1.03 to −1.84 at 8), positive at 9 on every seed
(jump +1.50 / +1.42 / +1.92), then 0.58 ± 0.15 at 12 and 0.76 ± 0.03 at 22; at points 4–6 its sign varies by seed. The
random network's cross-half is +0.27 at points 8–9 on seed 0 and negative at every point on seed 1, so this readout
cannot separate trained from random. The negative
values are a between-half miscalibration, not a mirror flip: at point 8 the cross-half MAE is 56° against 11° within a
half, where a left-right mirror of 8 directions would give 90°. On the paper-layout set the per-position onset is 6 and
cross-half transfer is already 0.81 at block 1, so the step on the hard set comes from the rendering. Caveats: per-position
features are averaged over the 8 time steps; the half-frame test is one pooled probe per half; the rendered sets sample
points 1, 4, 6, 7, 8, 9, 10, 12, 16 and 22 only (seeds 1–2 add 2, 3 and 5), so their onsets of 4 and 6 are upper
bounds; they reuse 7 start positions across all (θ, v) pairs; each hard set is 392 clips.

**Verdict.** On a harder stimulus, across three render seeds, one part of the paper's signature replicates: half-frame
transfer is negative at points 7–8 and positive from point 9 on every seed (paper layers 6–7 → 8). The sharp per-patch
rise does not: the largest rise is at points 4 → 6 on every seed, the 90% onset is 8 on two seeds and 9 on one, and the
random network's per-position curve also plateaus by point 6 (stratified folds; a start-grouped refit is in progress).
On the supplied clips neither part appears: the per-patch code forms by block 6 with no step at points 8 → 9, while the mean-pooled
curve is early under every fold grouping and at the paper's clip count. What training changes on both stimulus sets is
the per-position code from point 6 on (supplied: V-JEPA 2 0.96–0.98 against a mean of 0.39 for the random network;
hard: 0.81–0.97 against 0.57–0.62), not pooled availability; the half-frame dip is not separable from random, whose
transfer varies by render seed.

**Objective axis: VideoMAE** (v1 ViT-L, `MCG-NJU/videomae-large`, pixel reconstruction; 224-px input, 1,568 tokens; the paper used the VideoMAE-v2 family)[^obj]. VideoMAE matches V-JEPA 2
on every variable. Direction: block 1 0.886 vs 0.875, peak 0.992 (point 21) vs 0.991 (22), onset 2 for both. Speed:
peak 0.996 vs 0.994, onset 1. Acceleration: peak 0.996 vs 0.992, onset 1. Nested K at each model's peak is 67 / 109 /
87 for VideoMAE vs 88 / 89 / 67 for V-JEPA 2. Steering reaches the bar with 4 / 8 / 9 probes vs 4 / 7 / 6
(`figures/fig5_objective_axis.png`). Nothing in the pooled Part 1 measures on this stimulus is specific to latent
prediction; VideoMAE was not run per patch.

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
  random-feature picture from step 1. The counts depend on the coordinates: C.11 states no normalisation, and in raw
  centred coordinates (α re-chosen) nested K is 1.4–1.6× larger. Direction vs speed is then 51 vs 55 at point 9 and 65
  vs 73 at point 8, and under the paper-protocol rule 63 vs 61 and 68 vs 83[^raw]. So equal counts hold at point 9 and
  weaken at point 8, where speed needs more probes, not fewer.
- **The stop rule is not the same for both variables.** C.11 stops when either the R² rule or the MAE rule fires. For
  speed and acceleration the MAE rule (MAE > 0.9× the mean predictor's) fires first, at round R² 0.15–0.20 at every
  V-JEPA point under both protocols, while direction runs on to R² just under 0.1 (0.094 at the lowest). The scalar counts are therefore taken
  at a looser point than direction's. The same asymmetry means the speed and acceleration counts at Fig. 22's R² < 0.1
  (`K_loose`) are floors at every V-JEPA point except nested acceleration at onset, not only for VideoMAE as I
  wrote earlier (footnote [^fig22]). Rerun with one R² rule for all variables[^stop], which variable needs more probes
  depends on the rule and the protocol. Nested, the scalars need more: direction / speed / acceleration 37 / 47 / 48 at
  point 9 and 40 / 60 / 61 at point 8 at R² < 0.1, and 45 / 55 / 62 and 52 / 83 / 76 at R² < 0.05. Under the paper
  protocol direction and speed are about equal at point 9 (46 vs 50; 58 vs 59), and at point 8 direction needs more
  (83 vs 57; 113 vs 74), the one cell where speed needs fewer. The paper's claim holds only in that cell.
- **One column per round.** C.11 says "project out the learned direction", while each direction probe has two output
  columns. Removing one column per round (alternating sin/cos, or the top singular vector) takes 72–73 rounds nested
  and 84–92 under the paper protocol at point 9, about twice the stored 37 / 46, so the removed dimension count is
  about the same. Under ridge it creates no sawtooth (R² drop autocorrelation 0.82–0.84 nested, no isolated dips);
  under Adam the drops stay negatively autocorrelated, as with two columns, and one-column removal gives direction 1–2
  isolated dips (alternating: 1 on R²; top singular vector: 2 on acc15; two-column: 0) against speed's 2, so still no
  direction-specific sawtooth[^onecol].
- **Sawtooth.** Under ridge there is none. There are no isolated dips at any direction layer under either protocol.
  The lag-1 autocorrelation of per-round drops in R² is positive everywhere (0.42–0.95; a sawtooth gives negative
  values). On within-15° accuracy it is positive under the nested protocol (0.26–0.78) and mixed under the paper
  protocol (−0.24 at point 2, −0.06 at point 8, 0.27 at point 9, 0.19 at point 22). The paper's Fig. 23 teeth are about
  65 points deep by eye; here successive probes' readouts are 9–15° apart under ridge (mean consecutive readout angle,
  points 2–22), where a sin/cos pairing would put them near 90°[^saw].
- **Sawtooth on metrics both variables share** (8-bin accuracy and R², points 8 and 9)[^saw]. On 8-bin accuracy the
  drop autocorrelation under nested ridge is positive for both variables (direction 0.60 / 0.45, speed 0.68 / 0.74 at
  points 8 / 9); under the Adam recipe it is negative for both (direction −0.25 to −0.46, speed −0.38 to −0.50), and
  Adam's isolated dips number 1–3 per run for speed against 0–2 for direction. The "no isolated dips" statements for
  direction, here and in the Adam bullet, hold on within-15° accuracy only (on 8-bin accuracy the ridge paper-protocol
  run at point 8 has one). On a common metric speed is at least as jagged as direction.
- **Adam sequence (the C.11 recipe, run literally, targets standardised).** Direction reaches chance after K = 84
  probes (batch 64) or 96 (full batch) at point 9, and 88 or 100 at point 8. Speed takes 95 or 91 at point 9 and 104
  or 102 at point 8. That is about 2× the ridge paper-protocol K at point 9 (46 direction, 45 speed); for direction at
  point 8 it is 1.1–1.2× (ridge 83). Both variables turn jagged. The drop autocorrelation is negative for direction on
  within-15° accuracy (−0.30, −0.32 at point 8; −0.37, −0.34 at point 9) and for speed on R² (−0.44, −0.49; −0.54,
  −0.47). Direction has no isolated dips in any run; speed has 1, 2, 2 and 0. No round failed to train[^adam]. The
  literal Adam recipe makes both variables' curves jagged and their K larger, and neither curve has a
  direction-specific sawtooth. Both effects come from the optimiser, not the representation.

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

**Verdict.** The "tens of dimensions" claim reproduces at the paper's layer against a random band. The claim that
speed needs fewer probes does not reproduce under ridge (counts are equal; in the paper's own unit, dimensions, 2K for
direction, speed does need fewer in every cell), and direction's sawtooth does not appear under ridge. Under Adam both curves are
jagged, speed's as much as direction's on a metric both share, so the jaggedness tracks the recipe, not the variable.
The absolute counts depend on the coordinates (1.4–1.6× larger raw) and on which stop rule fires, which is looser for
the scalars.

### 3.3 Multi-probe subspace steering

**Paper's claim (Fig. 24).** One to a few probes barely move the readout, about 20 reach the target, and MAE-to-true
rises as MAE-to-target falls. **Figures:** `figures/fig3_steering_paper.png`, `figures/fig3c_steering_nulls_paper.png`,
`figures/fig3b_shift_heatmap_paper.png`.

Direction, target θ\* = 90°, evaluation probe fit on test (paper protocol; out-of-fold R² 0.967 at point 9, 0.980 at
point 22, 0.860 at point 2):

| N probes | pt 9 MAE-to-target | pt 9 rank-2K null, mean ± SD (p) | pt 9 rank-matched null, mean ± SD (p) | pt 9 MAE-to-true | pt 22 MAE-to-target (p) | pt 2 MAE-to-target (p) |
|---:|---:|---|---|---:|---|---|
| 0 | 87.4° | | | 2.3° | 87.3° | 86.5° |
| 1 | 78.4° | 77.7 ± 2.1° (0.67) | 72.6 ± 26.4° (0.48) | 9.1° | 71.7° (0.52) | 85.7° (0.62) |
| 3 | 25.4° | 30.0 ± 11.2° (0.33) | 66.5 ± 34.9° (0.095) | 62.0° | 12.9° (0.29) | 82.1° (0.57) |
| 5 | 8.7° | 12.8 ± 4.6° (0.14) | 57.2 ± 37.4° (0.095) | 78.8° | 5.5° (0.29) | 77.3° (0.62) |
| 10 | 3.1° | 9.0 ± 5.2° (0.095) | 54.3 ± 27.8° (0.048) | 85.4° | 4.4° (0.14) | 65.0° (0.48) |
| 20 | 2.9° | 14.2 ± 12.5° (0.048) | 62.5 ± 33.0° (0.048) | 87.3° | 2.8° (0.048) | 21.4° (0.57) |
| K (37 / 88 / 289) | 2.7° | 81.2 ± 42.4° (0.048) | (= rank 2K) | 87.0° | 4.1° (0.048) | 6.9° (0.048) |

p is the empirical rank against 20 random orthonormal bases (rank 2K, or rank-matched 2N), each with its own
least-squares solve (floor 1/21 = 0.048). The orientation-only null (learned coefficients through a random basis) gives
p = 0.048 at every N (mean 87.4° at N = 1). Source: `p1c_direction_L{2,9,22}.json`, `p1c_direction_L9_rankmatched.json`.

- **Null reading, two nulls.** Against the rank-2K (74-dim) basis the learned basis is no better up to N = 6 at point
  9 (p ≥ 0.14), first touches the 1/21 floor at N = 8 and stays there from N = 14 (point 22: from N = 17); at the onset layer (rank 578) it never beats it for N ≤ 20, so "single probe fails, many succeed" is
  largely least squares in *any* subspace of that rank. Against the rank-matched null (rank 2N for direction, N for
  scalars; 20 nested draws; second band in `fig3c_steering_nulls_paper.png`) the learned basis first beats all 20 draws
  at N = 6 (direction, point 9), N = 2 (direction, point 22), N = 6 / 7 (speed, points 9 / 19) and N = 5 (acceleration,
  points 9 and 21); at N = 1 no variable separates from either null (p ≥ 0.29). The rank-matched draws are heavy-tailed
  (direction SDs 23–42°; speed point 9, N = 7: 39.9 ± 165.2 m/s): a random low-rank subspace reaches the target only
  with a large edit (per-draw median norm ratio 10.6 on average, N = 1, direction point 9). At 70/30 the learned basis
  is at the 1/21 floor of the rank-2K null at N = 10 for all three variables (80/20: p = 0.095 / 0.14 / 0.095)[^s70][^rm].
- **Untrained network.** The same curve appears in the random-init ViT at point 9 (K = 24): 87.4° (N = 0) → 82.9°
  (N = 1) → 30.7° (N = 5) → 8.7° (N = 10) → 5.9° (N = 20)[^p1cr].
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
- **The evaluation-probe recipe** (point 9, direction)[^evp]. C.12 fits its evaluation probe on the test clips with
  R² = 0.99 and does not give its regularisation. My stored probe is ridge with α = 100 chosen by CV inside test
  (in-sample R² 0.993, the closest of the three to the paper's 0.99). A near-unregularised ridge probe (α = 1e-3,
  in-sample R² 1.000) and the C.11 Adam recipe (0.998) read the same edits differently: 3 and 4 probes reach ≤ 10° to
  target (stored: 5), the rank-2K null is beaten at the 1/21 floor from N = 5 for both (stored: p = 0.143 at N = 5,
  floor from N = 14), and the rank-matched null from N = 3 (stored: N = 6). The
  null reading therefore depends on how the evaluation probe is regularised, which C.12 does not fix.

**Verdict.** Fig. 24's shape reproduces with 3–4 probes under the near-unregularised and Adam evaluation probes and
5 under my CV-chosen one (C.12 gives no recipe; mine is the closest to its R² = 0.99), rather than 20. The probes beat
a random subspace of their own rank from N = 2–7 (N = 3 under the near-unregularised probes). Against a rank-2K
random basis they win from N = 5 under those probes but only from N = 14 under mine, and an untrained
network shows the same curve.

### 3.4 The paper's 70/30 split

Part 1 was rerun at the paper's literal 70/30 split (`splits/split_paper70.json`: same seed, stratification and
identical-clip grouping; `figures/split70/`)[^s70]. Every qualitative verdict is unchanged (`verdicts_changed: []`).

| 80/20 → 70/30 (train/test 1,200/300 → 1,050/450 for direction, 1,228/308 → 1,075/461 for the scalars) | direction | speed | acceleration |
|---|---|---|---|
| onset / peak point | 2 / 22 → 2 / 22 | 1 / 19 → 1 / 19 | 1 / 21 → 1 / 19 |
| nested K / paper-protocol K at point 9 | 37 / 46 → 36 / 35 | 39 / 45 → 36 / 40 | 41 / 47 → 38 / 47 |
| steering MAE-to-target at point 9, N = 1 / 5 / 10 | 78.4 / 8.7 / 3.1° → 75.3 / 6.7 / 3.3° | 0.69 / 0.16 / 0.08 → 0.51 / 0.08 / 0.08 m/s | 1.77 / 0.40 / 0.23 → 1.60 / 0.25 / 0.20 m/s² |

The paper-protocol K for direction drops from 46 to 35 because that count reads the test split at every round and the
test split is larger; the nested K barely moves. Part 2's contiguous design was also rerun at 70/30
(`results/split70_p2/`): the direction radius gap is +0.26 / +0.24 at points 12 / 22 (80/20) and +0.25 / +0.26 (70/30);
the point-22 endpoint loss of the spline (+3.80°) shrinks to +0.47°, and the point-12 endpoint gap is +0.11° and −0.40°
(both far below the probe's out-of-sample error). For speed and acceleration the bend is 0.04–0.19× centroid noise at
both splits, so the verdict flickers on sub-margin differences and "scalars are straight" holds at both. I kept 80/20 as
the split of record so that Part 1 and Part 2 read the same clips.

## 4. Part 2: spline steering

### 4.1 The circular structure

Recipe (Goodfire A.3): PCA-64 on train, one centroid per value, a periodic cubic spline for direction and a natural
spline for scalars. Goodfire's text takes the intrinsic angle as atan2(PC2, PC1) on the centroids, without labels, and its weekdays and
months 8B runs do so; only its 70B cyclic configs use the labels instead (§2). I compute a label-free angle, check it against θ, and use the
labels when the check fails.

| Direction layer | unsupervised angle vs θ: circ. corr (mean / max dev) | supervised circular chart MAE, radius | centroid PR / residual PR | expected sagitta ÷ centroid noise at 45° / 90° gap | LOO cubic beats line |
|---|---|---|---|---|---|
| 8 | failed in both planes → labels (flagged) | 7.1°, 3.17 | 3.75 / 7.99 | 0.15 / 0.59 | at 90° only |
| 12 | −0.983 (7.5° / 15.1°) | 3.5°, 5.56 | 3.41 / 7.84 | 0.22 / 0.85 | at 90° only |
| 22 | −0.964 (10.9° / 28.7°); pipeline plane (clips): 0.948 (11.8° / 34.1°) | 5.8°, 7.27 | 3.53 / 7.58 | 0.19 / 0.74 | never |

Source: `p2_geometry_direction_L{8,12,22}.json` (`angle.centroid`, `angle.activation`). The first figure in each cell
is the circular correlation between θ and the label-free angle atan2(PC2, PC1) in the centroids' top-2 PC plane. The
sign of −1 is an orientation flip, which is allowed. These figures use all 1,200 train clips. The steering runs fit on
the knot clips alone, and there the check decides the coordinate[^src]. At point 12 the centroid-plane angle passes on all
16 arcs and on the headline arc. At point 22 it fails on the headline arc (circular correlation 0.50 in the centroid
plane, 0.81 in the activation plane, order not preserved in either) and on 12 of the 16 arcs, which use the labels; the
other four (seeds 4, 8, 9, 10) passed and ran on the label-free angle, and I reran them on the labels (§4.3). Goodfire's
own label-free angle (each PC scaled by √variance, then their periodicity test) passes on 0 of 17 arcs at point 12 and
on 9 of 17 at point 22, where even the passing angles are up to 38–97° off θ (`p2_angle_goodfire_method.json`); run as
their code runs it, on all 64 centroids, it fails at points 8 and 12 and passes at 22 only, barely (relative
eigenvalue difference 0.445 against a tolerance of 0.45; circular correlation 0.905, max deviation 49.6°;
`p2_angle_goodfire_all64.json`). So
label-free recovery here is my extension, the centroid-plane fallback, and it works at point 12 only.

- **The ring is found** (`figures/fig4_centroid_plane_direction_L12.png`). The number to look at is the label-free angle
  above: circular correlation −0.983 with θ at point 12 and −0.964 at point 22 on all train clips; on the knot clips it
  holds at point 12 only. Separately, at point 12 the supervised
  chart and the centroid PC plane agree: the two angle assignments over the 64 centroids have circular correlation 0.994
  (mean deviation 4.7°). That number is agreement between two planes, not a correlation with the labels. The chart's
  principal angles to the top-2 PCs of the clip activations are 5.6° and 73.2°, so only one axis is shared: the ring is
  dominant among centroids, not among clips. On the labels knot order, held-out centroids on the contiguous arc are
  rebuilt with mean error 6.3 by the interpolating spline, 2.76 by the smoothing spline and 2.59 by the chord at point
  12, and 6.9 / 4.3 / 3.75 at point 22[^interp]; a larger interpolating error I reported earlier came from the knot order
  of the label-free angle, which is not monotone in θ, and is withdrawn. Steered on the labels angle, the interpolating
  spline keeps the path result (radius gap +0.28 on the headline arc at point 12, +0.30 at point 22), but its endpoint
  is worse at point 12 (+2.30° on the headline arc; +2.45° ± 2.14 over 8 arcs, 3 of 8 "negative_endpoint") and mixed at
  point 22 (−0.48° on the headline arc; +1.18° ± 1.57, 2 of 8), with an edit 1.4–1.6× the chord's on the headline
  arc at each point. All steering uses the count-weighted smoothing spline, which I chose on train folds, because the interpolating
  one rebuilds held-out centroids worse and makes a larger edit.
- **Circle, ellipse or bent line?** (`p2_ellipse_direction.json`, `figures/fig4g_ellipse_direction.png`). An ellipse.
  In the plane of the ring's own cos θ / sin θ component the axis ratio b/a is 0.74 / 0.87 / 0.74 at points 8 / 12 / 22
  by a direct conic fit, 0.74 / 0.82 / 0.72 from the full-space rank-2 chart, and 0.76 / 0.76 / 0.66 from the 2θ
  distortion of the label-free atan2 angle. Points 2–4 are much flatter (0.38–0.43) and point 10 is nearly round
  (0.89). Geometric residual in centroid-noise units at points 8 / 12 / 22: ellipse 0.92 / 1.15 / 1.11, circle 1.22 /
  1.55 / 1.36, smoothing spline 0.92–0.97 (about 1 by construction). So the ellipse sits at the noise floor and the
  circle does not. The ring is also bent out of its plane: a third centroid axis follows cos 2θ (a saddle) and holds
  0.28 / 0.20 / 0.24 of the centroid variance at 8 / 12 / 22, growing to 0.34 at point 14. At points 14–20 and 24 the
  saddle axis outranks the ring's minor axis, so a top-2 PC plane there is one ring axis plus the fold and shows a bent
  line; the ring itself is not one, and estimates made in that plane (eigenvalue ratio 0.91 at points 14–16 against
  0.66–0.77 in the ring plane) mislead. Planted controls in real point-12 activations recover a circle as 0.99, an
  ellipse of ratio 0.5 as 0.51, and a circle with a saddle of bend 1.2 as 0.99 in the ring plane but 0.84 by top-2
  eigenvalues. The random-init encoder's direction code is a cleaner ring: fold share ≤ 0.04, b/a 0.91–0.95 at every point (conic fit).
  Local curvature does not predict where the spline beats the chord: Spearman ρ between the ellipse curvature at the
  arc midpoint and the spline-minus-chord minimum-radius gap over the 15 distinct stored arcs (at point 22 four of
  them still on the label-free angle) is 0.05 (p = 0.85) at point 12 and −0.05 (p = 0.86) at point 22; the smoothing-spline curvature and the stored sagitta do no better (|ρ| ≤ 0.33, all
  p ≥ 0.23), and the curvature of ring-plus-fold reaches only −0.47 (p = 0.07) at point 22, with the wrong sign. With an axis ratio this close to one the spline's advantage is spread around the ring, not concentrated
  near the minor axis. PCA is the right first tool here: it finds the ring's plane, but its top-2 plane is not always
  the ring's plane.
- **Curvature vs noise.** Over any knot gap up to 45° the chord and the arc differ by less than a quarter of centroid
  noise. Held-out centroids on the contiguous 45° arc are rebuilt best by the chord at every direction layer (point
  12: chord 2.59, smoothing spline 2.76).
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
- **Is the ring occupied?** (780 held-out clips, chart plane, ring radius 1)[^p2b]. Yes, along its whole length: the
  largest angular gap between clips is 2.9° and neighbouring directions overlap (spread 2.2× the spacing at point 12,
  4.4× at point 22). The interior is empty except for the slowest speed-set clips (0.25–0.67 m/s: median radius 0.59,
  direction error 30°). Direction is undefined at zero speed, so a polar code should pull its slowest clips toward the
  centre. The midpoint of a 180° chord sits at radius 0.09, where 0% of held-out clips lie (point 22:
  0.14, 0.5%). This is what "dense manifold" means in the Goodfire paper, which defines a density metric (its Eq. 6) but
  never measures density.
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
curves, and a BF16 repeat. Each path has K = 50 waypoints (the §4.3 ring-occupancy and 5-NN side analyses recompute
the paths at K = 49 so that a waypoint sits at the midpoint). Held-out sources and targets take their coordinate by
linear interpolation of the labels between neighbouring knots (`Curve.coord_of_value`), for both arms alike, even in
the label-free point-12 runs. Every arm starts from the carrier's ground-truth value (an
oracle source coordinate; Goodfire instead starts every carrier at the centroid c_a whatever the carrier's own value,
A.6), so no arm has to infer where the carrier sits.

**The verdict rule is post hoc.** The rule that turns gaps into a call (`verdict.call` in each steering JSON) was
iterated during development, after looking at results. An early energy-based comparison was dropped as a deciding metric
once it was seen to reward residual erasure (the replace arms are on-curve by construction, so Eq. 9 favours them). The
final rule is symmetric: a gain or a loss counts only beyond the same practical margin, and an endpoint loss beyond that
margin overrides path gains ("negative_endpoint"). It was frozen at commit 8d3cac8, before the multi-arc sweep and the
70/30 reruns. The frozen rule was applied unchanged to all 32 arc runs and to every steering file cited here.

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
(258.75°–298.125°), so there are 15 distinct arcs in 16 runs. Point 12 runs on the label-free centroid-plane angle
throughout. At point 22, 12 runs used the labels angle and 4 (seeds 4, 8, 9, 10) the label-free one; I reran those four
on the labels (`results/arcs/L22_s{4,8,9,10}_labels/`), and the point-22 column is the all-labels set[^src]. Mean ± SD
across the 16 runs of the paired gap (spline − line, `gaps.manifold_minus_linear`):

| | pt 12 | pt 22 |
|---|---|---|
| min readout radius gap | +0.26 ± 0.05 (range +0.13 to +0.31) | +0.28 ± 0.03 (+0.23 to +0.32) |
| endpoint probe error gap | +0.08° ± 1.47 | +0.17° ± 1.33 |
| endpoint gap, duplicate arc counted once | −0.09° ± 1.35 | +0.16° ± 1.38 |
| verdict "negative_endpoint" | 0 / 16 | 1 / 16 |

The path result holds on every arc at both points, and the endpoint ties at both. At point 22 the one
"negative_endpoint" run is 191.25°–230.625° (+3.7°) and the other 15 are within ±3°, so the single-arc +3.80° above is
not typical. On the label-free angle the four rerun arcs had the two largest endpoint losses (+9.2° and +9.9°, the
duplicated arc) and the four smallest radius gaps (+0.11 to +0.19); on the labels they give +0.21°, +0.27°, −0.36° and
−0.25°, and radius gaps +0.26 to +0.32.

**Donor ceiling** (point 12 / 22, same carriers). Replacing the carrier's PCA-64 coordinates with those of a real
unseen test clip at θ\* reaches nearest-real R = 0.23 / 0.20, against 0.19 / 0.17 for both spline and line. Swapping
the whole activation adds almost nothing to R (+0.004 / +0.001). A real clip at θ\* reaches only R ≈ 0.2 against the
other real clips at θ\*, so this readout's ceiling is set by clip-specific variance, and the spline reaches 82% / 84%
of it. The MLP evaluator reads the in-subspace donor at 25.2° / 18.5° vs 30.7° / 21.2° for the spline[^donor]
(`figures/fig4d_donor_ceiling_direction_L12_L22.png`).

**All designs, endpoint probe error, spline vs line** (source `p2_steer_{var}_{var}_L{pt}_{design}.json`; extrapolation
column from `p2_extrapolation_linear_ext.json`[^ext]: the smoothing spline continued linearly along its end tangent, and
in brackets the authors'-code arm, an interpolating natural cubic continued the same way, with its own chord):

| Variable, point | scattered | contiguous | extrapolation (authors'-code arm) | held-out context |
|---|---|---|---|---|
| direction 12 | 7.00° vs 7.05° | 9.73° vs 9.62° | — | speed-set clips: 10.89° vs 10.92° |
| direction 22 | 5.86° vs 5.86° | 10.69° vs 6.89° | — | 11.08° vs 7.57° |
| speed 12 (m/s) | 0.156 vs 0.153 | 0.163 vs 0.133 | 0.195 vs 0.158 (0.207 vs 0.271) | |
| speed 19 | 0.083 vs 0.082 | 0.092 vs 0.080 | 0.247 vs 0.226 (0.220 vs 0.243) | |
| acceleration 12 (m/s²) | 0.345 vs 0.345 | 0.309 vs 0.305 | 0.739 vs 0.674 (0.413 vs 0.348) | |
| acceleration 21 | 0.279 vs 0.277 | 0.282 vs 0.266 | 0.761 vs 0.739 (1.021 vs 0.709) | |

Position sheet (speed set, start (x, y), thin-plate spline vs chord to a held-out interior 2×2 block): at point 12
"negative: TPS path indistinguishable from chord". At point 19 "chord better than the TPS path on err_path and
excess_to_nearest_real". Endpoint error is 0.178 m for both vs 0.197 for a Delaunay interpolation (point 12)[^sheet].

**Reading.** At the encoder layer the spline stays on the ring (the Eq. 9 distribution walks the arc in order) and the
line cuts across the ring's interior in the chart plane. The reflected arm is worst on every path metric, so it matters
which way the path bends. All of §4.1–§4.4 edits the pooled vector and reads it at the same point with probes and
distances, with no forward pass through the rest of the network; whether the model uses the edit is tested in §4.5. The radius gap grows with angular shift: at
point 12 it is −0.003 below 30°, then +0.05, +0.12, +0.25, +0.43 and +0.62 per 30° bin up to 180°, CI above zero in
17/17 runs from 90° on (16/17 at 60–90°; point 22: +0.02 to +0.66, 17/17 from 30°); endpoint error and nearest-real agreement show no
trend with shift[^p2b]. A readout that uses neither a probe nor the spline also separates the paths along the route:
the labels of the 10 real clips nearest the midpoint (PCA-64) are 16° / 21° (points 12 / 22) closer to the intermediate
direction for the spline than for the chord, 38° / 45° closer at shifts ≥ 120°, and agree with each other more
(resultant +0.12 / +0.11), on 17/17 arcs at both points. The empty interior is a ring-plane fact, though. In the 64-D
subspace the edit acts on, the spline midpoint is no closer to real clips than the chord's (5-NN distance ratio +0.007
at point 12, CIs split 4 above / 9 below) and at point 22 it is farther (+0.030, 17/17 arcs); in full space the stored
5-NN excess is +0.22 / +0.17 for the spline at points 22 / 12 (CI above zero in 17 / 7 of 17 runs; point-22 figures in this paragraph use the all-labels arc set,
`results/p2_shift_dependence_labels22.json`, `results/p2_ring_occupancy_L22_labels22.json`). So "the chord cuts
through the ring" holds in the ring plane only, and so does Goodfire's low-density-region premise here. For speed and
acceleration all arms coincide with the chord, extrapolation included: continued along its end tangent the smoothing
spline trails the chord by 0.02–0.07 on all four (all "path_geometry_positive"), and the authors'-code arm beats its
chord on speed and trails it on acceleration. The large extrapolation losses I reported earlier came from extending the
end cubic piece past the last knot, my choice, not the method's.

### 4.4 Controls and the comparison with Part 1

- **Random curves** (20 endpoint-matched draws, point 12 contiguous). Endpoint readouts match by construction. On the
  path, the spline ranks 1/21 on off-curve excess (0.15 vs a band of 0.68–0.88) and on Eq. 9 energy (0.92 vs 1.48–1.75).
  Unmatched random curves have endpoint error 87.9°. **BF16**: the winner on both energy metrics is unchanged.
  **Dose-matched line**: endpoint 9.41°, energy 1.27, radius 0.61, so the line's deficit is not a matter of dose.
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

At matched norm, Part 1's multi-probe subspace edit moves the linear readout furthest at point 12. Every arm leaves an
independent MLP evaluator at 19–39°, even where the linear probe reads 5–10°. A rank-2 ring rotation does not match the
higher-rank edits. **Strengths of the spline**: it closes around the circle and
keeps intermediate states on the ring in its plane (not closer to real clips in 64-D; §4.3). Its coordinate can be found
without labels only at point 12, through my centroid-plane fallback; Goodfire's own label-free angle fails its
periodicity test there, and point-22 steering and session 2 at points 2, 8 and 22 use the labels (§4.1). **Limitations**:
at held-out endpoints it offers nothing a chord does not; the interpolating version rebuilds held-out centroids worse
than the smoothing one and edits 1.4–1.6× more than the chord (hence smoothing); on scalars it adds nothing, and in
extrapolation it ties the chord once continued along its end tangent, as the authors' code does. **Failure cases**: the
point-22 endpoint on 1 of 16 arcs (+3.7°) and on the headline arc (+3.80°), and the position sheet.

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
  `shared_delta_fraction`) is 0.41 for the spline and 0.37 for the line at point 12, and 0.44 / 0.39 at point 22. More
  than half of every edit is clip-specific, so spline and line are not merely one constant shift plus a small
  correction.
- **Isometry with the straight-line baseline** (Goodfire A.5, behaviour = the predictor's forecasts)[^p2b]. Pearson r is
  0.86 / 0.96 / 0.69 at points 8 / 12 / 22 with distances along the smoothing spline and 0.80 / 0.90 / 0.78 with chord
  distances, reversed at point 22 on the stored coordinate (see below). The forecast and encoder-output spaces give the same r within 0.02 as the bare angle
  difference, so on a ring this test measures whether arc length is proportional to angle change. Goodfire's world-model
  behaviour manifold is built from activations (its Eq. 10), so its 0.996 isometry has the same circularity as our Eq. 9
  figures. With the authors' recipe (behaviour manifold = an interpolating spline through the per-value forecast
  centroids in the full 1,024-d forecast space) on the label-free knot order, r is 0.84 / 0.93 / 0.67 along the
  interpolating activation spline against 0.73 / 0.87 / 0.75 for the chord, and 0.885 / 0.979 / 0.758 against 0.800 /
  0.876 / 0.752 along the smoothing spline[^iso]. On the labels coordinate (Goodfire's 70B cyclic configs) the spline leads at every point: interpolating
  0.986 / 0.982 / 0.984 against chord 0.727 / 0.867 / 0.749 at points 8 / 12 / 22, smoothing 0.984 / 0.994 / 0.979
  against 0.858 / 0.890 / 0.831[^isol]. On Goodfire's own cyclic coordinate (the √variance-scaled atan2 its 8B weekdays
  and months runs use; its periodicity test fails at points 8 and 12 and passes at 22 by 0.005; circular correlation
  with θ −0.40 / 0.65 / 0.91) the chord leads at every point: interpolating 0.50 / 0.32 / 0.66 against 0.73 / 0.87 /
  0.75, smoothing 0.50 / 0.50 / 0.74 against 0.69 / 0.87 / 0.76[^isog]. So the isometry verdict is set by the knot
  coordinate, not by the curve: an ordering that follows θ makes the spline near-isometric to the forecast geodesics,
  and every ordering a label-free method gives us (our knot order, Goodfire's own angle) puts the chord ahead, at
  point 22 in particular. The point-22 reversal is therefore not withdrawn; it disappears only with the labels.

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

Edits at points 2–12 wash out: four blocks later the error is 82–88° for every arm but the overshooting spline, and
at point 25 MAE-to-true is back to 4–7°. The steered-point columns are read by a probe of the family that built the
edit, so they are not independent evidence. Point-22 edits survive the last three blocks (15–25° at point 25) with
weak target specificity: against the neighbouring target (at most 39° away, 16.5° on average[^s2]) the error rises by
only 1–6° at point 25 (7–13° at the steered point). The projection on the twin's real activation change at point 25
is 0.16 / 0.23 / 0.22 (probe-QR / smoothed spline / chord) vs 0.15 / 0.21 / 0.21 for the shuffled twin (`readout_b`).
The interpolating spline on the label-free knot order overshoots (‖Δ‖ 7.2× the natural twin change at point 12, vs
0.5–0.6×) and is excluded; on the labels order its edit is 0.97× the twin change[^il12].

**Predictor** (context frames 1–8 edited at every token; the predictor forecasts tubelets 4–7). *First attempt
(blind).* Probes fit on the encoder's real future tokens (3.4°, 6.7–8.4 px) read the unedited forecast 61° / 67.9 px
off[^fpos] (62.6° / 65.7 px when recomputed on the second attempt's test clips); token-space R (`session2_predictor.json`) is 0.0185 [0.0173, 0.0195] for the twin's context, ≤ 0.0025 for
every edit.

*Second attempt, a readout fit on the predictor's own outputs*[^nat]. Ridge probes (α by 5-fold CV) fit on the
unedited forecasts of the 480 probe clips read the 300 test clips' unedited forecasts at 8.9° (R² 0.92) and 11.8 px
(real-token probes: 62.6° / 65.7 px). The twin's own context now recovers R = 0.96 [0.92, 1.01] of the direction
change and R = 1.07 of the position change (9.1° from the target, 13.2 px from the twin), so the test can see an
edit. Edited forecasts, 200 carriers × 4 targets (unedited: 92.1°, 60.2 px); last column = median ‖Δ‖ at point 22
over the natural twin change (`session2_propagation.json`), not norm-matched (matched below):

| Arm | pt 12: to target | pt 22: to target | pt 22: to shuffled | pt 22: to 180° flip | pt 22: target − shuffled [95% CI] | pt 22: R dir / R pos | pt 22: px to twin | pt 22: ‖Δ‖ ÷ twin change |
|---|---|---|---|---|---|---|---|---|
| probe-QR | 84.1° | 17.1° | 23.5° | 162.9° | −6.4 [−7.3, −5.5] | 0.78 / 1.81 | 55.5 | 0.79 |
| radius-matched | 84.4° | 16.3° | 22.8° | 163.7° | −6.5 [−7.5, −5.6] | 0.79 / 1.49 | 43.2 | 0.58 |
| interpolating spline | 77.9° | 11.3° | 18.4° | 168.7° | −7.1 [−7.9, −6.3] | 1.00 / 1.43 | 44.3 | 0.91 |
| smoothing spline | 81.7° | 44.2° | 47.1° | 135.8° | −2.9 [−3.6, −2.2] | 0.60 / 1.28 | 38.6 | 0.62 |
| chord | 84.8° | 27.2° | 32.0° | 152.8° | −4.7 [−5.5, −3.9] | 0.68 / 1.30 | 39.7 | 0.65 |
| random, matched ‖Δ‖ | 92.3° | 85.7° | 85.4° | 94.3° | +0.3 [0.2, 0.4] | 0.10 / −0.05 | 61.1 | 0.79 |

At point 22 every structured arm moves the forecast toward the held-out target, away from its 180° flip; the weak
shuffled null (targets 16.5° apart) separates by 3–7°. Position overshoots: R 1.3–1.8, 39–55 px from the twin vs
13.2 px for its own context. At point 12 edits barely reach the forecast (81.7–84.8°); at points 2 and 8, 90.8–92.8°.
The point-12 interpolating-spline cell (77.9°) is the label-free knot order, whose edit is 10.7× the centroid change;
rerun on the labels order the edit is 1.40× the centroid change and the forecast reads 81.8° from target and 58.9 px
from the twin, like every other point-12 arm[^il12]. So the point-12 failure is wash-out, not knot order. The 180°-flip
column equals 180° minus the target column by construction; only the position flip (distance to the twin reflected
about its tubelet-0 centroid) is an independent check. Unmatched, the spline's lead over the chord
(11.3° vs 27.2°) is confounded with a larger edit (0.91× vs 0.65×; the point-12 contrast and flip null are not).
*Norm-matched rerun*[^nm]: point-22 edits rescaled per (carrier, target) to a common norm, same carriers, targets and
probes (chord and unedited forecasts match the cache exactly). Cells: to target / paired target − flip / R dir / px:

| Arm | at the chord's norm (0.65× twin change) | at the natural norm (1.0×) |
|---|---|---|
| probe-QR | 23.2° / −134° / 0.64 / 51.4 | 18.8° / −142° / 1.04 / 69.8 |
| radius-matched | 17.7° / −145° / 0.92 / 48.1 | 23.3° / −133° / 1.56 / 65.6 |
| interpolating spline | 19.5° / −141° / 0.68 / 43.3 | 12.3° / −155° / 1.21 / 43.7 |
| smoothing spline | 39.1° / −102° / 0.63 / 39.0 | 37.3° / −105° / 1.25 / 43.5 |
| chord | 27.2° / −126° / 0.68 / 39.7 | 25.7° / −129° / 1.37 / 43.3 |
| random | 86.4° / −7 [−20, 6] / 0.10 / 61.2 | 84.2° / −12 [−24, 1] / 0.12 / 63.0 |

The ordering survives matching and the margin shrinks: the interpolating spline lands 7.7° closer than the chord at
the chord's norm (paired 95% CI 5.0–10.6) and 13.4° at the natural norm (11.1–15.6), against 15.9° unmatched: about
half of that lead was dose, and a route effect remains. The gain is angle-only: at the chord's norm its R equals the
chord's (−0.003 [−0.04, 0.03]) and position is 3.6 px worse [2.5, 4.7]; at the natural norm R is 0.155 lower [0.06,
0.26] and position ties (+0.4 px [−1.2, 2.1]). The smoothing spline trails the chord by 12° at both norms.

**At the encoder output**[^enc]. Point 25 is the final LayerNorm, whose tokens are the predictor's input and the site
Goodfire §5 steers in its world-model experiment. The same 200 carriers × 4 targets and six arms, with the edit added
to every post-LN context token, the same predictor-native probes, and each arm at its own norm, at the chord's norm
and at the natural twin change:

| Arm | own norm: to target / R dir / px to twin | at the chord's norm | at the natural norm |
|---|---|---|---|
| chord | 12.5° / 0.81 / 42.5 | (same) | 18.0° / 1.78 / 65.9 |
| interpolating spline | 15.7° / 0.87 / 44.4 | 23.4° / 0.62 / 41.2 | 15.8° / 1.23 / 52.6 |
| radius-matched | 13.8° / 0.79 / 57.1 | 22.7° / 0.70 / 50.0 | 16.2° / 1.25 / 87.2 |
| probe-QR | 22.0° / 0.82 / 105.4 | 49.8° / 0.41 / 53.2 | 27.0° / 0.73 / 90.3 |
| smoothing spline | 27.3° / 0.74 / 38.5 | 23.8° / 0.75 / 39.5 | 31.1° / 1.65 / 64.6 |
| random | 109.4° / −0.29 / 81.1 | 99.9° / −0.13 / 65.5 | 108.0° / −0.26 / 76.7 |

Unedited: 92.1°, 60.2 px; the 180° flip null of the structured arms is 153–168° at each arm's own norm (130–168° across the three norm
conditions; ≈ 165° for the best arms), again
180° minus the target error. Edits at the predictor's input reach the forecast as point-22 edits do, so the wash-out
picture holds: edits at points ≤ 12 are repaired, edits at point 22 or later reach the forecast. The route effect does not carry over.
Paired spline − chord is +3.1° [2.1, 4.2] at own norms, +10.8° [9.4, 12.6] at the chord's norm (point 22: −7.7° [−10.6,
−5.0]) and −2.1° [−3.8, −0.4] at the natural norm (point 22: −13.4°). "The spline beats the chord at matched norm" is a
point-22 result; at the encoder output the chord is as good or better. For comparison with Goodfire: its §5 world-model
evidence is qualitative, decoded frames along one linear and one manifold path between two car positions
(`refs/steering_paper.txt` l.1850–1874). A probe readout over 800 carrier-target pairs is a stronger test of the same
claim, not an analogue of theirs.

**The forecast read along the path**[^ap]. Nine waypoints (t = 0, 0.125, …, 1) per point-22 path, scaled so the
endpoint equals the natural twin change (t = 1 reproduces the cells above); random: a line with the spline's norms.
Cells: error to the ideal intermediate direction (source + t × shorter-arc shift) / forecast radius / px to the twin:

| Arm | all 800 pairs: t = 0.25 | 0.5 | 0.75 | 1.0 | shift ≥ 135° (216 pairs): t = 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| interpolating spline | 13.8° / 0.95 / 53.5 | 17.8° / 0.98 / 47.7 | 21.8° / 1.08 / 44.0 | 12.3° / 1.34 / 43.7 | 17.4° / 0.88 / 75.0 | 23.4° / 0.81 / 65.2 | 32.7° / 0.83 / 56.7 | 12.1° / 1.24 / 54.7 |
| smoothing spline | 12.1° / 0.95 / 51.4 | 17.9° / 0.96 / 48.4 | 28.0° / 0.96 / 45.5 | 37.3° / 0.97 / 43.4 | 15.9° / 0.95 / 74.4 | 18.4° / 1.06 / 71.4 | 32.5° / 1.07 / 61.8 | 47.1° / 1.01 / 52.5 |
| chord | 22.0° / 0.72 / 46.7 | 38.9° / 0.63 / 38.9 | 35.6° / 0.78 / 38.2 | 25.7° / 1.03 / 43.3 | 41.9° / 0.53 / 64.4 | 79.4° / 0.36 / 51.3 | 60.1° / 0.61 / 48.5 | 31.5° / 1.01 / 54.8 |
| random | 28.3° / 1.00 / 63.0 | 49.7° / 1.05 / 65.9 | 69.2° / 1.12 / 69.7 | 84.4° / 1.18 / 73.3 | 46.1° / 0.98 / 88.9 | 79.1° / 1.04 / 91.2 | 107.5° / 1.13 / 94.6 | 126.2° / 1.19 / 97.0 |

Mean error to the ideal intermediate direction over t: interpolating spline 16.3° [14.7, 18.1], smoothing spline 22.1°,
chord 29.7° [26.4, 33.1], random 53.2°; paired spline − chord −13.4° [−15.7, −11.0], −31.3° [−36.0, −26.7] at shift
≥ 135°; smoothing − chord −7.6° [−10.2, −5.0]. The spline's forecast moves steadily through the intermediate
directions; at large shifts the chord's stays near the source until about t = 0.5, then jumps (largest step 68°, spline
57°), and its radius collapses (minimum 0.26 vs 0.53 for the spline at ≥ 135°, 0.36 vs 0.81 at t = 0.5). Caveats: edit
size is matched only at the endpoint (spline 0.40, chord 0.25 of the natural change at t = 0.25); along the path the
chord is closer to real clips in full space (1.21 vs 1.32× the 5-NN floor), while in the ring plane, at shifts ≥ 135°, it dips to radius
0.69 and the spline stays ≈ 1.26; mid-path position favours the chord (38.9 vs 47.7 px at t = 0.5) because the twin is
the endpoint; at exactly 180° (12 pairs) the ideal direction follows the spline's arc. The intermediate-state claim now
comes from the model's own forecast, not from the geometry of the edit.

**The reverse test**[^ap]. For 20 carriers × 2 targets at 135–180°, I optimised K = 9 waypoints in PCA-64 at point 22
so the forecast follows the ideal intermediate direction (Adam, 100 steps, converged by 20; edit norm ≤ 1.2× natural;
gradients through blocks 23–24 and the predictor in bf16, re-scored in fp32). They hit it to 0.08°, but only at the
norm cap, with 28% of each edit in the ring plane and forecast radius ≈ 5 (unedited ≈ 0.96). The path radius goes
2.2 → 1.0, never crossing the interior; cosine 0.12 with the spline's edit, 0.00 with the chord's; closer to the spline
in the ring plane (1.35 vs 1.75), slightly closer to the chord in full space (+0.04 [0.02, 0.06]). With an angle-only
objective the predictor can be steered along an off-ring route, so the reverse direction is not recovered here. This
objective is weaker than Goodfire's full pullback objective, which also penalises leaving real behaviour. A negative.

**Time-reversed clips** (the forward-trained probe read on reversed clips)[^trev]. From point 1 on, the direction probe
reads θ + 180° on the reversed clip: the error to θ + 180° is 20.5° at point 1 and 5.7–10.3° from point 2 on, with
98–100% of clips closer to the flipped angle. The probe tracks motion direction, not position or occupancy (a reversed
constant-velocity clip has the same frame set). A speed probe transfers to reversed clips (R² 0.964–0.981 vs
0.936–0.987 forward at points ≥ 1).

## 5. Beyond the three variables

| Question | Result | Source |
|---|---|---|
| Object permanence | Direction decoded from time steps whose frames contain no disk (89 clips; test 15 clips / 22 tokens): MAE 7.5° [5.6, 9.4] at point 8 (visible 5.4°), 6.1° at point 22 (visible 3.8°); shuffled-label null 84.5°, p = 0.001. The random-init encoder, same clips and protocol, does as well: test-clip absent-step MAE 5.9–7.2° across points vs 6.1–13.4° for V-JEPA 2 (null ≈ 90° for both). V-JEPA 2 is ahead only late, by ≤ 1.0° on test clips (points 16–25) and 2.4° / 2.0° pooled at points 22 / 25; it is behind at points 1–12. On visible steps V-JEPA 2 is 5–7° better from point 8 on. So above-null decoding after the disk leaves is attention mixing within the clip (no causal mask), not learned carrying. | `p1a_object_permanence.json` (`random_init.side_by_side`), `fig6_object_permanence.png` (random-init overlaid) |
| Cartesian vs polar | On constant-velocity clips (596), (vx, vy) reaches onset at point 1 and (sin θ, cos θ) at point 2 (difference −1, CI [−1, −1]); block 1 R² 0.929 vs 0.863. Speed set: 0.985 vs 0.855. The one-block "emergence" of direction is the normalisation v/‖v‖. Direct test at block 1: the angle of the (vx, vy) probe's output has MAE 12.1° against 12.3° for the direct (sin, cos) probe, and R² 0.900 against 0.911 once the direct output is scaled to unit length, so the direct probe's lower R² there is its radius, not its angle; the two angles disagree clip by clip by 13.1°. From point 2 the direct probe is better (8.3° vs 11.2°). | `p1a_support_onset_*_meanpool.json`, `fig1d`, `p1a_support_cartesian_angle.json` |
| Direction transfer (held-out context) | Direction probe fit on the direction set, read on the speed set at point 9: MAE 4.4° (source CV 4.0°); 8.7° below 1 m/s, 3.3° at 1–4 m/s. On the acceleration set: 5.8°. At point 1: 10.8° (23.9° below 1 m/s). | `p1a_support_transfer_meanpool.json`, `fig1c` |
| Spatial generalisation | Train on start x < 0, test on x > 0, point 9: R² 0.971 (MAE 4.9°), vs 0.972 within-side. At point 22, mean-pool 0.957 vs disk-pool 0.988. | `p1a_support_spatial_{meanpool,diskpool}.json`, `fig1e` |
| Direction vs speed subspace (paper C.4 method) | Overlap direction←speed 0.0740 at point 8 (random expectation 0.0781, 5–95% band 0.0756–0.0808); 0.0733 at point 9 (0.0723, band 0.0694–0.0740). Direction vs acceleration 0.0762 and 0.0739, inside or at the edge of the band. The INLP bases are as orthogonal as random ones, yet steering direction still moves the speed readout (§3.3 off-target). | `step2_subspace_angles.json` |
| Objective axis | V-JEPA vs random-init at the direction peak: probes needed to reach ≤ 10° MAE 4 vs 10; nested K 88 vs 26. VideoMAE matches V-JEPA 2 on all three variables with the same onsets (§3.1): 4 probes to the bar, nested K 67, peak 0.992. | `objective_axis.json`, `fig5_objective_axis.png` |
| Position sheet | Start (x, y) is decodable; 36-cell centroid PR 7.46 (point 12) / 3.76 (point 19), Procrustes to (x, y) 0.38 / 0.66; spline steering gives no path advantage (§4.3). | `p2_sheet_speed_L{12,19}.json` |

## 6. What this says about her framing

- **Detection vs use.** Detection is easy on this data. A random network, and random features of a 32-number
  trajectory, detect all three variables at R² ≥ 0.85. Use is where the evidence thins. The Part 1 edit moves a
  held-out linear probe, but whether it beats a rank-2K random basis depends on the evaluation probe (from N = 5
  under the paper's, N = 14 under mine; a rank-matched one from N = 2–7), it works in an untrained network too, and it
  leaves an MLP on disjoint clips 19–31° off. On the ladder in `PART2_RATIONALE.md` §2, this project reaches rung 3 at
  the steered layer and rung 4 only by circular measures. Rung 5 was tested: at point 22 and at the encoder output the
  edit reaches the predictor's forecast; at points ≤ 12 it washes out and does not (§4.5). At point 22 the forecast
  passes through the intermediate directions on the spline and jumps on the chord (−13.4°); at the encoder output the
  chord lands as close as the spline or closer.
- **Internal world model vs stimulus-response, and which readout to trust.** For this stimulus class the random-init
  control shows that pooled linear availability is architectural and that carrying direction into disk-free tokens is
  not a training effect. The per-patch probes show what training does add: the random network never gets past a mean
  per-position R² of 0.39, the regime of fragmented local signal that pooling adds up, while V-JEPA 2 reaches 0.96 by
  block 6 on the supplied clips and, on a harder stimulus, forms that code with its largest rise at points 4 → 6 on every render seed and, across the paper's depth, loses
  and recovers half-frame transfer (§3.1). So the emergence zone is a claim about the per-patch readout, which the paper's C.5 says, and a mean-pooled
  curve can neither confirm nor refute it. VideoMAE matches V-JEPA 2 on every pooled Part 1 measure (I did not run it
  per patch), so none of this is specific to latent prediction or shows that the variables are used to predict. The authors' OpenReview response states that "all 13 models encode motion direction
  (R²≥.43), regardless of objective", classification CNNs included, so availability is their own finding; training buys
  precision, fewer probes to steer (4 vs 10 to reach 10°[^obj]) and a label-free ring.
- **The linear representation hypothesis: right about the subspace, wrong about the moves.** Direction lives in a 2-D
  linear subspace (sin, cos) with a ring on it, and nothing in Part 1 contradicts that form of the hypothesis. The
  steering corollary is what fails geometrically: in the ring plane the straight path between distant directions
  crosses the empty interior (readout radius 0.61) where the curved path does not (0.86), though in the 64-D edit
  subspace it is no farther from real clips (§4.3). Independent readouts at the steered layer do not care; the
  predictor's forecast at point 22 does, in angle only, and at the encoder output it does not (§4.5). The supported statement is "for a cyclic variable the
  hypothesis describes the subspace and misdescribes the moves, geometrically".

## 7. Limitations and next steps

- **Stimulus.** A single disk on a flat background is nearly pixel-decodable. The per-patch step at the paper's depth
  appears only on the hard rendered set (§3.1), which has 392 clips and 8 directions, covers direction only, uses one
  render seed and reuses 7 start positions across all (θ, v) cells, so it is a reproduction on one small set, not on
  the supplied data.
- **Per-patch probes.** Features are averaged over the 8 time steps at each position, so time structure within a
  position is not probed; the half-frame test is one pooled probe per half; the rendered sets were read at 10 points
  only, so their onsets of 4 and 6 are upper bounds. VideoMAE and speed were not run per patch.
- **Training dynamics.** With intermediate V-JEPA 2 checkpoints, the random-init vs final contrast becomes a curve.
  That is the natural test of when precision and the ring appear.
- **Predictor readout is a probe; the edit overshoots.** A probe of the pooled forecast, not a rendered future;
  point-22 edits overshoot in position (R 1.3–1.8); one stimulus, four targets in one 45° arc. The spline's lead is in
  angle only and the smoothing spline (the Part 2 default) is the worst real arm.
- **The hollow is in the ring plane.** In the 64-D edit subspace and full space the chord is no farther from real clips
  than the spline (§4.3). Along the path the forecast follows the intermediate directions along the spline and jumps
  along the chord (−13.4° paired, −31.3° at large shifts); the reverse test does not recover the ring (§4.5).
- **Post-hoc verdict rule.** Iterated after seeing results, frozen at 8d3cac8 before the arc sweep (§4.2); the gaps and
  CIs are the evidence.
- **Label-free coordinate.** Found only at point 12 and only through my centroid-plane fallback; Goodfire's own
  label-free angle fails its periodicity test at point 12, and point-22 steering and session 2 at points 2, 8 and 22 use
  the labels, as only Goodfire's 70B cyclic runs do (§4.1).
- **Encoder-output steering.** At point 25, the predictor's input and Goodfire's site, edits reach the forecast
  (chord 12.5°, interpolating spline 15.7°) but the spline's matched-norm lead over the chord from point 22 does not
  carry over (+10.8° worse at the chord's norm, 2.1° better at the natural norm; §4.5). The route effect in the forecast
  is a point-22 result.
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
  `session2_extras.py`, `session2_native_readout.py` (predictor-native probes), `session2_norm_matched.py`, `session2_along_path.py` (forecast along the path, reverse test; `session2_predictor_along_path.json`, `session2_reverse_path.json`). The 70/30 reruns set `WM_SPLIT_PATH=splits/split_paper70.json`. Second look at Part 2 (commit 33a2cbd, clean): `run_ring_occupancy.py`, `run_isometry_linear.py`, `summarize_shift_dependence.py`; `PART2_SECOND_LOOK.md` is the audit record against the Goodfire paper. Part 1 follow-ups: `p1a_perpatch.py` (GPU extract + per-patch probes; `p1a_perpatch_direction_*.json`), `run_paperscale_velocity.py`, `run_support_cartesian_angle.py`, `run_audit_robustness.py --items 1 2 3 4` (grouped CV, raw coordinates, sawtooth metrics, evaluation-probe recipe), `run_stop_rules.py --items 1 2` (stop-rule sweep and one-column removal). Part 2 follow-ups: `session2_encoder_output.py` (point 25 and the point-12 labels-order spline), `run_angle_goodfire.py` (Goodfire's periodicity test, `--no-holdout` for all 64 centroids), `run_isometry_linear.py --angle labels`.
- **Provenance.** Every results JSON records the split sha256 (`98e6310c…`), seeds (split 0, all others 0), git
  commit and a dirty flag. The 20 `p2_steer_*` files and all 32 arc runs were produced at commit 8d3cac8; the other
  Part 2 files at 677b305, 8e552c1, 8829195, 8f08444 or fbf4f72; all with `git_dirty_src_or_scripts: false`. Most
  Part 2 files record split and source paths inside the frozen scratchpad worktree that ran them; the split sha256 and
  the commit are the same as the repository's. The session 2 files were scored at b9c53d0 or 0f34ec2 (the native
  readout at 8734f4b, the norm-matched rerun at 494afe2, along-path and reverse at 21b27b6) with the dirty flag set;
  the encoder-output and point-12 labels-order files at 3f4c8de (forward at b7d09fc), clean. The per-patch files record
  commit 4413937; the grouped-CV, raw-coordinate, sawtooth and evaluation-probe files 46a33da, the velocity-only and
  Cartesian-angle files 6fc2529, the labels isometry and all-centroid angle files 7de664a, all clean.
- **Numerics.** CPU–GPU parity on 8 clips: worst per-layer max|Δ|/max|x| 8.2e-5 (rule < 1e-3); GPU batch-8 vs
  batch-16 gap 1.31× the CPU–CPU gap (rule ≤ 2×). Frame hashes and disk masks match, and 27/27 sha256 checks of the
  downloaded activations pass[^gpu].
- **Cost.** GPU session 1 (RTX 4080 SUPER, Vast): 37.7 billed minutes, $0.19[^gpu]. Session 2 box (RTX 4060 Ti,
  $0.198/h): $0.512 to the end of session 2 (the run itself $0.311); $1.19 over 6.00 billed hours as of 22:44 ET, box
  still running, including the native-readout extraction (0.09 h, $0.018), the norm-matched rerun (0.137 h, $0.027),
  the along-path forward (0.34 h, $0.067) and the reverse test (0.365 h, $0.072); $0.006 egress[^cost]. The per-patch
  run on the same box took 1.65 h ($0.33; box totals not refreshed since), and the encoder-output forward 817 s of GPU
  time with no separate cost recorded (`session2_encoder_output.json`, `forward.seconds_total`). No other box's cost is recorded.
- **Tests.** `pytest --collect-only` collects 202 tests at the commit of this report.

[^gpu]: `artifacts/gpu_session1.json`.
[^ptxt]: Line numbers in `refs/physics_paper.txt` (text of arXiv 2602.07050): (a) 1211 vs 1172; (b) 1207 vs 1243 and 1266; (c) 1245–1246; (d) 430–431 vs 1256 and 1269; (e) 669 vs 1243; (f) 245 vs 671; (g) 1244 vs 1256 and 1269, Table 3 at 958, 1217 vs 970–973, 402–403; (h) Table 3 rows 949–951. Our numbers: `p1b_*` (`K`, `K_loose`), `results/split70/COMPARISON.md`, `p1c_direction_L9_strict.json` (`strict_eval`), `p1c_direction_L9.json`.
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
[^fig22]: `loose_threshold` in each `p1b_*` file: R² < 0.3 for direction and R² < 0.1 for speed and acceleration, the thresholds of the paper's Fig. 22. For speed and acceleration the K at that threshold equals the nested K at every V-JEPA point except acceleration at onset (493 vs 466). `K_loose_censored` is set only in the VideoMAE files (true for acceleration at point 22, where K_loose = 87 is a floor); the V-JEPA files predate the flag, and there speed and acceleration K_loose is also a floor at every point (the MAE rule stops the sequence while R² is still 0.15–0.20) except nested acceleration at onset, where R² reaches 0.071 (`rounds[].cv_r2`, `paper.rounds[].test_r2`).
[^stim]: `results/stimuli/{paper_layout,hard}/p1a_direction_direction_meanpool{,_random}.json`; `results/session2_stimuli_validation.json`.
[^s70]: `results/split70/comparison.json`, `results/split70/COMPARISON.md`, `results/split70_p2/COMPARISON.md` and `results/split70_p2/p2_steer_*_contiguous.json`; 80/20 null p from `results/p1c_{direction,speed,acceleration}_L9.json`.
[^tr]: `results/p2_two_route_direction_L12_L22.json`.
[^rot]: `results/p2_rotating_speed_axis_L12_L22.json`.
[^donor]: `results/p2_donor_ceiling_direction_L12_L22_contiguous.json`.
[^cos]: `results/p2_cosine_tangent_{direction_L12,direction_L22,speed_L12,speed_L19,acceleration_L12,acceleration_L21}_contiguous.json` (`cosines.step_net`, `cosines.tangent_chord`).
[^trev]: `results/session2_timerev.json`, `results/session2_timerev_speed.json`.
[^fpos]: `results/session2_future_position.json`.
[^rm]: `results/p1c_{direction,speed,acceleration}_L{9,22|19|21}_rankmatched.json` (`rank_matched_null`: `rows[].rank_matched`, `rows[].random_basis_full_rank`); `figures/fig3c_steering_nulls{,_paper}.png`.
[^nat]: `results/session2_predictor_native_readout.json` (`direction`, `position`, `twin_context_reference`, `per_layer`, `definitions`); `scripts/session2_native_readout.py`.
[^nm]: `results/session2_predictor_norm_matched.json` (`per_condition`, `spline_minus_chord`, `session2_unmatched`, `applied_norms`, `forward.parity_*`); `scripts/session2_norm_matched.py`.
[^p2b]: `results/p2_ring_occupancy_L{12,22}.json` (`occupancy`; `over_arcs.manifold_minus_linear.{knn_mid_err,knn_mid_rbar,mid_ratio_pca64}`), `results/p2_shift_dependence.json` (`by_shift.{probe_radius_min,probe_err_to_target,nearest_real_R,excess_to_nearest_real}`), `results/p2_isometry_linear.json` (`correlations["smooth.{geo,lin}.{predictor,encoder_out,concept}"]`); 17 arc runs (seeds 0–16); scripts at 33a2cbd.
[^cost]: `results/session2_cost.json` (`cost_start_to_session2_done_usd`, `session2_run_cost_usd`, `cost_so_far_usd`, `billed_hours_so_far`, `as_of_utc` 02:44 UTC, `runs[0..3]`, `pull_egress_usd`, `status: running`).
[^ap]: `results/session2_predictor_along_path.json` (`by_t.{all,shift_ge_135}`, `summary`, `spline_minus_chord`, `applied_norms`, `n_exact_180`; unedited radius `by_t.all.spline.forecast_radius[0]`) and `results/session2_reverse_path.json` (`reverse`, `by_t`, `over_t`, `paired_over_t`, `loss_curve_mean` every 10 steps); `scripts/session2_along_path.py`.
[^src]: `results/p2_angle_source_audit.json` (`angle_source_counts`, `headline_angle_source`, `headline_L22_angle_check`, `L22_labels_reruns`, `L22_aggregate_all_labels`; the duplicate-once figure is computed from its per-arc rows), `results/arcs/L22_s{4,8,9,10}_labels/`, `results/p2_angle_goodfire_method.json` (`goodfire_passes_periodicity_test`, `goodfire_angle_vs_labels`), `results/session2_plan.json` (`per_layer[].angle_choice`); Goodfire's coordinate source: `refs/causalab/causalab/configs/analysis/activation_manifold.yaml` (`intrinsic_mode: pca`, inherited by `runners/{weekdays,months}/*_8b_pipeline.yaml`), `runners/{alphabet,age}/*_8b_pipeline.yaml` and `runners/*/*_70b*.yaml` (`intrinsic_mode: parameter`), `methods/spline/train.py:59` (default `parameter`).
[^interp]: `results/p2_interp_labels_summary.json` (`heldout_reconstruction.labels_order.contiguous`, `seed0_interp_labels`, `arcs_interp_labels`, `delta_ratio_spline_over_chord`), `results/p2_interp_labels/`, `results/arcs_interp/`.
[^ext]: `results/p2_extrapolation_linear_ext.json` (`runs.*.{smoothing_linear_ext,interp_linear_ext_goodfire_code,stored_smoothing_cubic_ext}`), `results/p2_linear_ext/`, `results/p2_linear_ext_interp/`.
[^iso]: `results/p2_isometry_goodfire_method.json` (`layers.{8,12,22}.new.{interp,smooth}.{geo,lin}_pearson`).
[^isog]: `results/p2_isometry_goodfire_coord.json` (`layers.{8,12,22}.goodfire_angle.{interp,smooth}.{geo,lin}_pearson`, `goodfire_periodicity_test`, `angle_vs_labels`, `geo_below_chord_goodfire_angle`; 213 tests at ee828a7).
[^isol]: `results/p2_isometry_goodfire_labels.json` (`layers.{8,12,22}.labels_angle.{interp,smooth}.{geo,lin}_pearson`; `unsupervised_angle` rows reproduce the label-free figures; `geo_below_chord_labels_angle` false at every point).
[^pp]: `results/p1a_perpatch_direction_{vjepa2,vjepa2_constvel,random,vjepa2_hard,vjepa2_paper_layout}.json` (`curves.{perpos_mean_r2,pooled_mean_r2,pooled_frac_ge_0.5,cross_half_r2,meanpool_r2}`, `onsets.*`, `layers[].halves` for the cross-half MAE, `methods`, `provenance.time_averaging`); `figures/fig1g_perpatch_direction.png`, `fig1h_perpatch_heatmaps.png`; rendered-set layout (7 shared starts) in `results/session2_stimuli_validation.json` (`layout.start_rule`) and `scripts/render_hard_stimuli.py`.
[^seeds]: `results/p1a_perpatch_hard_seeds.json` (per seed `layers[].perpos.mean_r2`, `halves.cross_r2`, `onsets`, `largest_jump`), `results/p1a_perpatch_direction_vjepa2_hard_seed{1,2}.json`, `results/p1a_perpatch_direction_random_hard{,_seed1}.json`, `figures/fig1j_perpatch_hard_seeds.png`; render seeds at 2abb3e9, extraction on the box from a frozen worktree (GPU forward 611 s), 3a8d7c7.
[^gcv]: `results/p1a_grouped_cv.json` (`sets.{direction,speed}_{vjepa2,random}.{stratified,direction_grouped,start_grouped,speed_grouped}.{onset,onset_ci}`); `figures/fig1i_grouped_cv.png`.
[^psv]: `results/p1a_paperscale_velocity_only.json` (`summary`, `models.vjepa2.n392_velocity.onset_per_seed`).
[^raw]: `results/p1b_raw_coordinates.json` (`cells.{direction,speed}_L{8,9}.{raw,stored_zscored}.{nested_K,paper_K}`).
[^saw]: `results/p1b_sawtooth_metrics.json` (`cells.{direction,speed}_L{8,9}.{ridge_nested,ridge_paper,adam_b64,adam_full}.stats.{r2,bacc8,acc15}`); consecutive readout angles from `sawtooth.mean_consecutive_angle_deg` in `results/p1b_direction_direction_meanpool_L{2,8,9,22}.json`; the paper's teeth read from its Fig. 23 by eye.
[^evp]: `results/p1c_direction_evalprobe_recipe.json` (`stored`, `recipe_results.{ridge_alpha_1e-3,adam_c11}.{eval_probe,n_to_10deg,random_basis_p,rank_matched_p}`); the stored floor N from `random_nulls.rows[].random_basis.empirical_p_to_target` in `results/p1c_direction_L{9,22}.json`.
[^enc]: `results/session2_encoder_output.json` (`per_arm.*.{unmatched,chord_norm,natural_twin_norm}`, `spline_minus_chord`, `spline_minus_chord.point22_stored`, `unedited`); `scripts/session2_encoder_output.py`.
[^il12]: `results/session2_interp_labels_L12.json` (`spline_labels`, `stored_point12.spline`, `stored_spline_edit_norm_recomputed`).
[^stop]: `results/p1b_stop_rules.json` (`cells.*.{nested,paper}.{K,stored_stop_trigger,r2_at_stored_stop}`, `common_r2_stop_points_8_9`).
[^onecol]: `results/p1b_one_column_removal.json` (`cells.direction.{two_column_stored,one_col_alternate,one_col_top_sv}.{ridge_nested,ridge_paper,adam_b64,adam_full}`).
