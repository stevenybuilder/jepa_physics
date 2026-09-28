# V-JEPA 2 physics take-home: report

Every number below comes from a file in `results/` (including `results/split70`, `results/split70_p2`,
`results/arcs`, `results/stimuli`) or `artifacts/gpu_session1.json`, named in a footnote or in the table's source line.
Paper claims are stated qualitatively and attributed to the paper.

## 1. Summary

The README asks for a small-scale reproduction of Joseph et al. (arXiv 2602.07050) in the frozen V-JEPA 2 ViT-L/16
encoder, for direction, speed and acceleration, and an open-ended Goodfire spline-steering extension (arXiv 2605.05115).

**Three defended claims.** (1) The per-patch code is training-selective: on the supplied clips V-JEPA 2's mean
per-position R² is 0.96 at block 6, while the random-init network pools to 0.86–0.88 at every depth but its mean
per-position R² never exceeds 0.39; on a harder rendered set the half-frame jump replicates at the paper's depth on all
three seeds, a shape the random-init network never shows. (2) Every variable needs tens of probes at the paper's layer,
far outside a random-removal band, and Step 3 reproduces in shape and, with C.11's Adam probe sequence, the basis C.12
steers along, in count: it takes 18 to reach 10°, where the paper reports ≈ 12° at about 20. (3) Direction lies on a
ring, and at held-out direction values the spline path stays on the ring while the straight path cuts across the hollow
in the ring plane (in the 64-D subspace it is as close to real clips at point 12 and closer at point 22):
the minimum readout radius is higher for the spline on every arc (point 22 on the labels angle[^src]).

**Two disagreements with the paper.** (1) On the harder set transfer does not appear only after the zone: it is 0.7 at
point 1, and the per-position curve rises most between points 4 and 6 on every seed, though under the 90%-of-max rule
its onset is 9 / 8 / 8, her depth, and the sigmoid inflection 5.7–5.9 (§3.1): partial agreement on a harder stimulus,
not on the supplied one. (2) Whether speed needs
fewer probes than direction depends on the stop rule and the probe recipe more than on the network. Under ridge on the
same clips speed needs fewer under C.11's thresholds (7 of 8 cells in probes, 8 of 8 in the paper's unit, dimensions),
and the paper's plotted Fig. 22 sits with that rule for speed (≈ 28 at layer 8, where Fig. 23's speed curve ends near
R² 0.05) but not clearly for direction (Fig. 23's layer-8 direction sequence runs to about 100 probes while Fig. 22
plots 44, so that count reads as an earlier cutoff, one of the two readings the paper allows, since C.12 itself reports 25 probes under R² < 0.1 at layer 8; read with the caption's rule for direction and C.11's for speed, speed
needs more on the same clips, 11 vs 26 and 19 vs 28 at point 9); under the caption's thresholds (R² < 0.3 / 0.1) speed needs more in all 8 cells, across the two supplied
sets the counts are equal at point 9 and the peaks (speed fewer only at point 8 under the paper protocol, 45 vs 83), and
under the literal Adam recipe the stored sets give speed at least as many in 3 of 4 cells, while on the same clips Adam
repeats ridge's pattern (speed fewer under C.11 at points 8, 9, 19 and 22: 110 vs 131, 81 vs 103, 92 vs 140, 68 vs 198;
more under the caption's)[^adamsc]. The firm disagreement is
the sawtooth: neither the ridge curves nor the paper's literal Adam recipe give a direction-specific one. **One addition to its steering result.** An edit along one probe's axis
fails (78° with the ridge probe, 84° with the Adam probe; §7.2 says > 80°); an edit built from that probe but weighted by the activation covariance, which leaves the
probe's plane, reaches 3.2°, and so does the same construction on a random 2-D subspace (median 4.2°, p = 0.24), so what
the learned probe buys is specificity, not target error.

**One new thing.** Edits at points ≤ 12 wash out within a few blocks and barely reach the predictor's forecast; point-22
edits survive to the output, and along the point-22 path the forecast follows the intermediate directions along the
spline and jumps along the chord (−13.4° paired, −31.3° at large shifts). This is a probe of the forecast on one
stimulus and one 45° arc, not a rendered future. The wash-out holds at the edits' own norms, and speed shows it too. Scaled to the
natural twin change, though, a point-12 speed edit moves the forecast's speed readout 0.57 of the way to the target,
so for speed it is partly a matter of dose. Direction at point 12 was not rerun at that norm (§4.5).

**One negative.** At held-out endpoints the paper's comparison baseline, the chord between the raw centroids (A.9), run
with our matched-support edit, lands closer than the smoothing spline: 4.7° against 9.7° at point 12 and 3.6° against
10.7° at point 22 on the headline arc (clip-bootstrap gap +5.0° [3.9, 6.1] and +7.0° [6.0, 8.1]). The headline arc is the
extreme case at point 12 and second-largest at point 22 (one arc, seed 16, reaches +8.3°): over the 16 held-out arcs the raw chord's lead is +1.3° ± 1.8 SD at point 12, with the spline ahead on 3,
and +2.3° ± 2.0 at point 22, with the spline ahead on none. The spline ties only the chord between its own smoothed knots, which was our line arm until
the parity audit[^rawchord]. Speed and acceleration are straight, and there the spline
adds nothing; in extrapolation our smoothing spline, continued along its end tangent as the authors' code does, trails
the chord by 0.02–0.06, while the authors' own interpolating arm beats the chord on speed and trails it on
acceleration[^ext].

Rerunning Parts 1 and 2 (Part 2: the contiguous steering runs) at C.12's 70/30 split (C.11 states 80/20; App. B uses 5-fold CV) changes no qualitative
verdict; §2 lists every deviation from the paper and what the parity audit changed.

## 2. Setup

- **Model.** `facebook/vjepa2-vitl-fpc64-256` (transformers 4.56.2), frozen. Input is 16 frames at 256², decoded
  with PyAV, divided by 255 and normalised with ImageNet mean/std. There is no resize or crop (`src/wm/extract.py`),
  giving 8×16×16 = 2,048 tokens. Forward passes run in fp32 with TF32 off for matmul and cuDNN (flags recorded as
  false[^gpu]). There are 26 hidden-state points: the embedding, blocks 1–24 (the raw output of block 24, captured by
  a hook), and the final LayerNorm. The paper's "layer L" is our point L+1, so the paper's layer 8 is point 9.
- **Representation.** `meanpool`, the mean over all 2,048 tokens, is used for every Part 1 result, as in the paper.
  `diskpool` (tokens the disk covers) and `timepool` (per time step) are used for controls.
- **Data.** direction 1,500 clips (64 angles, a mix of constant-velocity and accelerating-from-rest clips); speed
  1,536 (64 speeds, 24 clips each); acceleration 1,536. The README is explicit: *"The supplied dataset is
  deliberately smaller and simpler than the datasets in the paper. The aim is to reproduce the methodology and
  qualitative findings, not the paper's exact numerical results."* The stimulus is one orange disk on a flat
  background with a fixed camera. Acceleration clips all start at rest, so acceleration, mean speed and displacement
  are one variable here. The paper's acceleration set has the same confound, and I did not try to fix it, so the introduction's claim (l.121–123, echoed by Table 1) that acceleration "can be approximated directly by a single MLP" without a velocity intermediate is untestable on either dataset.
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
| Probe fit | linear probe, 20-config lr × wd sweep, 5-fold grouped CV, mean ± SD across folds (App. B; the optimiser is named only in C.11: Adam) | ridge, α by CV | parity check below: ridge ≥ Adam on all three variables at the peak; App. B's sweep run at points 0–10 gives direction onset 2 [2, 3] and speed onset 1 [1, 1], the same as ridge, with the sweep 0.0015–0.074 below ridge at every point (coupled L2 weight decay, at the grid's low edge for 10 of 12 direction points and 5 of 12 speed points)[^appb] |
| Input | 224², 1,568 tokens | 256², no crop, 2,048 tokens | layer fractions comparable, patch counts not |
| Hidden states | 24 points | 26 (embedding, blocks 1–24, final LN) | paper layer L = our point L+1 |
| Data | 8 directions, one velocity set (8 θ × 7 v × 7 starts) read for both direction and speed | 64 directions, mixed motion types | direction also reported per motion type |
| Split | 80/20 (C.11 l.1207) with stopping on test; 70/30 only in C.12 (l.1243); 5-fold grouped CV in App. B | 80/20, stopping on fold-mean CV | "paper protocol" also run; Part 1 and Part 2's contiguous steering runs rerun at 70/30, no verdict changes (§3.4); session 2, isometry and pullback not rerun |
| INLP K | probes until test at chance | nested K (held-out folds), paper-protocol K beside it | both below |
| Steering basis length | until R² < 0.1 on train | ridge: all-train sequence cut at nested K; Adam refit (§3.3): K = 84 by C.11's stop rule read on the test clips | ridge length never chosen on test; the Adam length is, which touches N = K and the rank-2K null only |
| Steering solve (C.12 l.1235 "least squares") | c* via least squares such that all probes predict θ* | minimum-change c* = c + A⁺(y* − ŷ) in the full rank-2K basis at every N | a literal minimum-norm solve in V_K would also erase the clip's other 2K − 2N coordinates; the two coincide when V is built from the first N probes; the erase reading was not run |
| INLP recipe | Adam lr 1e-3, wd 1e-4 | ridge; literal Adam sequence run once at points 8 and 9 | curves jagged and K 1.1–2.3× (≈ 2× except direction at point 8) under Adam; no direction-specific sawtooth (§3.2) |
| Attentive-MLP probes | §3.2: patch-preserving attentive-MLP probes, "interpret both jointly" with the pooled ones | not run; the per-patch ridge probes of C.5 are the patch-preserving readout here | none on the claims tested (the attentive results in the paper are IntPhys) |
| INLP coordinates | no normalisation stated (C.11), raw features implied | train-z-scored features | raw centred coordinates raise nested K 1.4–1.6×; direction vs speed equal at point 9, not at 8 (§3.2) |
| INLP stop rule | R² or MAE rule, whichever fires (C.11) | the same rules, with C.11's undefined "random baseline" for speed read as the fit-set mean predictor (a shuffled-prediction reading would leave speed to its R² rule alone; at nested point 8 that gives speed 83 against direction's 80 dimensions) | the MAE rule stops speed and acceleration at R² 0.13–0.20, direction runs to R² just under 0.1 (§3.2) |
| Onset | "one-third depth", no numeric rule | first sampled point at ≥ 90% of the maximum, 200-draw clip bootstrap | per-patch curves reported beside the pooled ones (§3.1) |
| CV folds | "5-fold grouped" (App. B), key not stated | stratified by value; direction-, start- and speed-grouped folds rerun | onsets unchanged (§3.1) |
| Per-patch probes | per patch (C.5), method not stated | one probe per spatial position on features averaged over the 8 time steps; a pooled-patch probe; the half-frame test is one pooled probe per half | time structure within a position is not probed |
| Rendered sets | start sampled per (θ, v) pair (App. A) | 7 starts drawn once and shared by every (θ, v) cell | 7 distinct start positions, not up to 392 |
| Steering coordinates and clips | C.12 states no normalisation; steers the velocity set at 8 directions (l.1251) | every edit and null is minimum-norm in train-standardised coordinates (`src/wm/steer.py`), and all 300 direction-set test clips (64 angles, half accelerating) are steered; no raw-coordinate or velocity-only steering run | untested: the covariance-weighted result (§3.3) shows the metric decides the single-probe number, so the 78° and the small-N null p-values are z-score-space figures |
| Steering evaluation probe | fit on test, R² = 0.99 (C.12) | ridge on test, α = 100 by CV inside test (in-sample R² 0.993) | α = 1e-3 or Adam need fewer probes and beat the rank-2K null sooner (§3.3) |
| Objective axis | VideoMAE-v2 family | VideoMAE v1 ViT-L (`MCG-NJU/videomae-large`), 224² | "not the objective" is shown for v1 only |
| Part 2: steering site | Goodfire: last-token residual stream (A.2); encoder output for the world model (§5) | mean-pool over 2,048 tokens at point L | the edited vector is not one the model consumes; §4.1–§4.4 read it with probes, §4.5 adds the edit to every token |
| Part 2: PCA-64 fit set | all prompts in the task (A.3) | knot clips (folds 0–2) at the kept values only | held-out values never shape the subspace; the plane can differ from an all-clip fit (point 22, §4.1) |
| Part 2: spline | interpolating, through the centroids exactly (A.3); √count-weighted smoothing spline for the world model (B.1) | smoothing spline with weight √count / sd_c per knot and coordinate and s = number of knots (both my choices; B.1 gives no smoothing value), fit by FITPACK `splrep`, which chooses its own knot subset per coordinate (a regression spline), where the authors' code uses a Reinsch penalty with a knot at every centroid and one λ; interpolating run beside it | interpolating rebuilds held-out centroids worse and its edit is 1.4–1.6× the chord's (§4.1) |
| Part 2: direction coordinate | unsupervised atan2(PC2, PC1) (A.3; the weekdays/months 8B configs inherit `intrinsic_mode: pca`); ordinal index for the sequential tasks (A.3; alphabet/age configs `parameter`); the labels only in the 70B cyclic configs | our centroid-plane atan2 at point 12; labels at points 2, 8 and 22 | label-free only at point 12, and only through our fallback (§4.1) |
| Part 2: manifold arm | replace the PCA-64 part with the curve point (A.6) | additive, x + γ(t) − γ(t_src), residual kept | theirs run as a labelled arm (§4.4) |
| Part 2: base pair of arms | manifold vs whole-activation chord replacement (A.6) | spline vs chord in the same PCA-64 subspace (matched support, additive; causalab ships the same support as its non-default `linear_subspace` mode, in replacement form) | their default linear arm erases the residual; run separately and labelled (§4.4) |
| Part 2: waypoints | K = 50 (A.6; the weekdays/months 8B default); alphabet/age 8B configs 150/250 (alphabet_8b_n3 50), 70B configs 100–150, grid/cylinder 20 | K = 50 | E_BC sums over waypoints, so only within-run energy ratios compare |
| Part 2: energy aggregation | per prompt, summed over waypoints, mean over prompts per pair, mean ± SE over pairs (A.7) | quoted energies are flat means over all (carrier, target) rows; the per-pair version is stored beside them (`over_pairs`) | arm ordering identical; absolute values differ by ≤ 0.06 (point-12 spline 0.92 vs 0.93 per pair, dose-matched line 1.40 vs 1.44) |
| Part 2: Eq. 10 temperature | τ = 0.5 on a LayerNorm'd 64-d latent (B.1) | τ = 0.5 in raw PCA-64 units | absolute energies not comparable; τ 0.25–2 keeps the point-12 ordering (`tau_sensitivity` in `p2_steer_direction_direction_L12_contiguous.json`) |
| Part 2: behaviour manifold | smoothing spline through 128 bin centroids (B.1); Eq. 10 bin centres are the activation-manifold spline itself evaluated at B evenly spaced positions (B.1) | interpolating spline through the 64 per-value centroids in the Hellinger tangent plane (A.4), F over 128 bins; the Eq. 10 bin centres lie on a separate labels-coordinate smoothing spline fit on the probe folds, not on the M_h the arms walk | circular at the steered layer either way (§4.2); our Eq. 10 reference is less circular than B.1's, and the point-22 near-tie (§4.3) is measured against it |
| Part 2: carriers | 16 fixed base prompts per task, one set for every pair (A.6) | 48 test clips per target, each steered from its own value | Goodfire starts every carrier at the centroid c_a whatever the carrier's own value (A.6); ours starts each carrier at its true value (an oracle source) and averages over sources |
| Part 2: pullback (A.8–A.9, C.3) | one path per (source, target) pair, loss averaged over 16 carriers (A.8); replace the top-32 PCs, other PCs and residual held; L-BFGS; path = natural cubic through 10 free control vectors evaluated at K = 20, chord init (A.8), 20 free points with kNN-graph init in the causalab config, K = 30 all free with 30 pairs for the world model (C.3); squared-Hellinger target on M_y; norm regulariser off for weekdays only (5·10⁻⁴ months, 10⁻³ age); scored by closest-point residual and intrinsic R² (A.9), and for the world model by mean distance to M_h (C.3: chord 2.22, geometric 0.20, pullback 0.29) | one path per carrier; additive edit in PCA-64; Adam, zero init, 8 free waypoints; 1 − cos loss on the forecast angle; hard cap 1.2× natural change; equal-t distances | rerun with their recipe (8 pairs, 32 evaluations per pair, unconverged) with all geometry from the edited context-only activations: from an on-ring start the path moves to 0.87 from both spline and chord (tie), about five times the chord's 0.16 from the spline, with the carrier mean 0.14 off the ring and individual carriers 0.45–0.54; neither protocol recovers the ring (§4.5) |

**What the parity audit changed.** A paper-first audit of our own methods moved eight verdicts:

- Part 1, emergence zone: read per patch on the hard render across three seeds, the 90%-of-max onset is 9 / 8 / 8 (her depth) while the largest rise is at points 4 → 6 on every seed, and the half-frame dip-and-jump at points 8 → 9 (paper layers 7 → 8) is training-specific (§3.1).
- Part 1, Step 3: with our refit of the paper's C.11 Adam probe sequence as the steering basis the probe count reproduces, 18 probes to 10° and 16 to the paper's 12° at about 20 (§3.3).
- Part 1: a covariance-weighted edit built from one probe steers to 3–5°, outside the probe's plane, as does the same construction on a random 2-D subspace (median 4.2°), so the learned probe adds specificity, not reach (§3.3, §6).
- Part 2: the isometry verdict is set by the knot coordinate; under every label-free ordering point 22 is a tie (spline only under the basic interval in the fully faithful run), and Goodfire's own angle loses to the chord at point 12 (§4.4).
- Part 2: at the encoder output the verdict is mixed (the spline trails the chord by 10.8° at the chord's norm and leads by 2.1° at the natural norm), and in scalar extrapolation the spline trails the chord by 0.02–0.06 (§4.5, §4.3).
- Part 1, Step 2: the paper reads direction and speed off one velocity set and our stored counts came from two supplied sets; rerun on the same 750 constant-velocity clips under ridge, "speed needs fewer probes" holds in 7 of 8 cells under C.11's thresholds, which the paper's plotted Fig. 22 follows for speed (≈ 28 at layer 8) though not clearly for direction (Fig. 23's layer-8 direction sequence runs to about 100 probes against Fig. 22's 44), and in none under the caption's, so the verdict is a stop-rule call (§3.2).
- Part 2: our line arm joined the spline's smoothed knots; the paper's comparison baseline is the chord between raw centroids (A.9), which sits 1.9–3.7 PCA units from the spline point at the held-out targets and beats the spline at the endpoint by 5.0° and 7.0° on the headline arc and by 1.3° averaged over the 16 point-12 arcs, so "endpoints tie" became "the paper's chord wins the endpoint" (§4.3, §4.4).
- Part 2: Goodfire's cyclic 8B runs take the coordinate label-free, as atan2(PC2, PC1), and only its 70B cyclic configs and two 8B weekdays demo configs use the labels (§2).

**Places where the paper contradicts itself**[^ptxt]. (a) The INLP stopping
threshold for direction is R² < 0.1 in C.11 and R² < 0.3 in the Fig. 22 caption, and for speed 0.05 against 0.1: each
`p1b` file has both (`K`, `K_loose`; §3.2). (b) The split is 80/20 in C.11 and 70/30 in C.12: I ran both (§3.4). (c) C.12 fits its evaluation
probe on the test clips it then steers and scores (independent of the steering probes but not of those clips): her
protocol is our §3.3 headline, with a split-half version beside it (12.4° / 17.1° at N = 5 vs 8.7°). (d) The main text
reports < 0.5° with all probe directions, the appendix 11.9° held-out with 20 probes: I report every N (all probes: 2.7° ridge, 2.9° Adam). (e) The velocity
set has 392 videos in App. A, but C.12's split is 240 + 103 = 343. (f) The main text measures motion "in pixels per
frame", App. A in m/s. (g) Probe counts disagree across C.10–C.12: C.12 trains "25 probes until R² < 0.1" at layer 8
and reports 20; Table 3 gives a layer-8 direction dimension of 136 (68 probes); C.11 says 14–136 while Table 3 lists 400
at layers 20–23; the main text says 40–50, up to 80. (h) "Speed needs fewer" depends on the layer in the paper's own
Table 3: at layers 0–2 the direction dimension is 30 / 30 / 14 (15 / 15 / 7 probes) against speed's 25 / 24 / 25, so
speed needs more probes there, and fewer from layer 3 on. Goodfire's paper
has a smaller one of its own: A.3 derives the cyclic coordinate as atan2(PC2, PC1) "in an unsupervised manner", and
the weekdays and months 8B configs, the paper's cyclic runs, do inherit that mode (`intrinsic_mode: pca` in
`configs/analysis/activation_manifold.yaml`); the sequential tasks use the ordinal index as A.3 says (alphabet and
age configs `parameter`). The coordinate's only text-vs-config gap is the 70B weekdays and months configs, which set `parameter`,
the labels, for a model the paper's one-dimensional experiments do not report (A.2: 8B layer 28 "for all tasks").
A.6 says K = 50 waypoints, which the weekdays/months 8B and alphabet_8b_n3 configs use, where alphabet/age 8B use
150/250, the 70B configs 100–150 and the grid/cylinder configs 20. A.8 parameterises the pullback path as a natural
cubic through 10 free control vectors, while C.3 says that "following the language-model setup, all K + 1 waypoints
(including endpoints) are free parameters", with K = 30; the released configs run weekdays with
free points, a kNN-graph start and no norm term (`configs/analysis/pullback.yaml` defaults), and only alphabet 8B with
A.8's spline, linear start and 5·10⁻⁴ norm weight.

**Probe-recipe parity** (point = CV-peak layer; pooled out-of-fold R², targets standardised for Adam[^recipe]):
ridge vs Adam (C.11 recipe) is 0.9905 vs 0.9858 for direction, 0.9940 vs 0.9882 for speed and 0.9925 vs 0.9871 for
acceleration. The best coupled-L2 Adam grid cells reach 0.9868, 0.9907 and 0.9887 (AdamW 0.9870, 0.9905, 0.9889).
Every ridge value lies above the Adam CI, so ridge is slightly *better* on all three variables.

## 3. Part 1

### 3.1 Layer-wise probing

**Paper's claim.** Speed and acceleration are decodable early. Direction appears only from about one third of the
depth (the "Physics Emergence Zone"). The representation of physical variables "peaks in the middle layers, and degrades toward the output" (introduction; the abstract says the same, and
§4.2 makes the claim for IntPhys). App. C.5 places the
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
the CI. Source: `p1a_{var}_{target}_meanpool.json`.

**Mean-pooled curve.** Direction is at 0.875 after one block, and no variable declines late on the pooled readout (per patch it does: the per-position mean peaks at 0.976 at points 10–14 and is 0.939 at point 24, the pooled-patch probe goes 0.977 → 0.922 and half-frame transfer 0.960 → 0.826, while the random-init network climbs to 0.385 with no late decline (0.007 → −0.021 over points 0–2, then rising with dips of ≤ 0.01 at points 12–14 and 16–19), so "degrades toward the output" partly reproduces per patch). The paper's §5.2 also says Cartesian velocity
and acceleration "exhibit a transition at the Physics Emergence Zone"; the same passage adds that acceleration is "also decodable with
high R² from early layers"; on pooled probes here (vx, vy) reads 0.985 / 0.977 / 0.984 and (ax, ay) 0.975 / 0.966 / 0.980
at block 1 / point 8 / point 9, so no transition shows in the pooled readout. Disk-pooling changes little
(direction peak 0.994, onset still 2[^disk]). The onset does not depend on how the CV folds are grouped (the paper's
App. B says "5-fold grouped" without the key): direction onset is 2 [2, 2] with stratified, direction-grouped, start-grouped and
8-sector-grouped folds (block 1 fold-mean R² 0.875 / 0.847 / 0.869; the sector figure, 0.828, is pooled out-of-fold R²,
since a one-sector fold makes per-fold R² meaningless), and speed onset 1 [1, 1] with stratified and
speed-grouped folds[^gcv]. Nor does it depend on the probe recipe: App. B's 20-config Adam sweep at points 0–10 puts
the direction onset at 2 [2, 3] (point 2 clears the 90% threshold by 0.005) and speed at 1 [1, 1][^appb]. This is a disagreement with the paper's §5.3 and Fig. 2c, where pooled direction "becomes reliably decodable only at the
Physics Emergence Zone" and reads about 0.2–0.6 before it: mean pooling is her primary readout (§3.2), and C.5 predicts only
"modest" pooled performance before the zone, not 0.875. C.5 does put the sharp change in the per-patch readout, so that is
the second test, below.

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
selective (the random network stays flat). This does not explain the missing emergence zone:
at the paper's own clip count the onset comes back to point 2, both for 392 constant-velocity
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
| supplied, random-init ViT-L | −0.01 / 0.22 / 0.27 / 0.29 / 0.38 | 16 [10, 19] | 0.65, 4 [3, 5] | 0.61 / 0.70 / 0.71 / 0.74 | 1 [1, 1] |
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
R² 0.5 from point 7 on every seed, 0.996 of them at point 6 on seed 1); across points 8 → 9 it rises +0.093 / +0.065 / +0.065, so the 90%-of-max onset is
9 [9, 9] on seed 0 and 8 [8, 9] on seeds 1 and 2; pooled-patch onset on seed 0 is 9 [8, 9]. A random-init network on
the same clips reaches 0.57 at point 6 and 0.60–0.62 from point 8, so the trained network leads by 0.21–0.35 from point
6 on and not at point 4, where random is ahead (0.54 vs 0.52; at point 1 it is behind, 0.41 vs 0.49); on seed 1 the
random curve rises 0.36 → 0.61 with a 90% onset of 7. Cross-half transfer
is 0.69–0.73 at point 1, negative at points 7 and 8 on every seed (−1.03 to −1.84 at 8), positive at 9 on every seed
(jump +1.50 / +1.42 / +1.92), then 0.58 ± 0.15 at 12 and 0.76 ± 0.03 at 22; the dip begins at points 4–5 (seed 1: −1.16 at 5, +0.22 at 6, −1.03 at 8, so its 5 → 6 recovery of +1.37 nearly
matches its 8 → 9 one; seed 2: −0.43 at 5). The recovery at 9 is only to about chance (0.08 / 0.39 / 0.08); the
point-1 level returns by point 22. The random network shows no dip: on seed 0 its cross-half rises monotonically from
−0.72 at point 1 to +0.27 at points 8–9 (8 → 9 change −0.01), and on seed 1 it stays negative throughout (−0.21 at 9,
8 → 9 change +0.22). So the dip-and-jump shape is training-specific; the level at point 9 is not. The negative
values are a between-half miscalibration, not a mirror flip: at point 8 the cross-half MAE is 56° against 11° within a
half, where a left-right mirror of 8 directions would give 90°. On the paper-layout set the per-position onset is 6 and
cross-half transfer is already 0.81 at block 1, so the hard set's dip comes from the rendering (each render seed changes the 7 starts and the floor texture
together, so seed-to-seed swings cannot be assigned to either). Caveats: per-position
features are averaged over the 8 time steps; the half-frame test is one pooled probe per half; the rendered sets sample
points 1, 4, 6, 7, 8, 9, 10, 12, 16 and 22 only (seeds 1–2 add 2, 3 and 5), so their onsets of 4 and 6 are upper
bounds; they reuse 7 start positions across all (θ, v) pairs; each hard set is 392 clips. C.5's "marked increase in
redundancy" was not measured; only spatial spread (share of positions above R² 0.5) was.

**Verdict.** On a harder stimulus, across three render seeds, one part of the paper's signature replicates: the half-frame
jump. Transfer dips to −1.0 to −1.8 at point 8 and jumps back to about chance at point 9 on every seed (paper layers
7 → 8), a shape the random network lacks; the paper's stronger claim, that probes "begin to generalize to unseen regions
only after the emergence zone", does not hold here, since transfer is already 0.7 at point 1. The sharp per-patch
rise does not: the largest rise is at points 4 → 6 on every seed, the 90% onset is 8 on two seeds and 9 on one, and the
random network's per-position curve also plateaus by point 6. A 5-fold refit on seed 0 gives per-fold onsets 8, 9, 9,
8, 8 on the stored stratified folds and 9 in all five folds when whole start positions are held out (per-position
0.17 → 0.71 → 0.72 → 0.89 at points 4 / 6 / 8 / 9, so the largest rise is still 4 → 6); the cross-half sign change
across 8 → 9 appears in 5 of 5 folds under both schemes[^hfolds]. Under the sigmoid criterion the paper's rebuttal
proposed, as summarised in our notes (fit R² > 0.9, inflection at ≤ 50% depth, peak ≥ 15 pp above chance; we read the
R² as the fit's, depth as (point − 1)/23 and chance as R² 0, none of which is the only reading), fitted on the sampled
points (1–22 for the hard and paper-layout sets, 10 or 13 points; 1–24 for the supplied set), the hard-set per-position curves inflect at points 5.73 / 5.76 / 5.93
(depth 0.21; fold bootstrap 5.36 [4.78, 5.74], start-grouped 4.66 [3.79, 5.50]), the supplied set at 2.10 and the
paper-layout set at 3.52. The cross-half curves inflect at her depth (8.95–9.02, depth 0.35) but fail the fit test
(R² 0.24–0.48); the mean-pooled speed and acceleration curves also pass with inflections at or just past her depth (9.76;
10.17, 1.2 points past), on fitted rises of 1–2 points from a baseline R² of 0.97–0.98 (fit R² 0.93–0.94), which is the criterion's weakness with chance set at zero: it also accepts
the random-init per-position curves (supplied set: inflection 3.26, peak 38.5 pp) and the random-init hard-set
cross-half curve on seed 0, along with curves whose inflection sits at the first point and one random-init mean-pooled
curve with a negative fitted rise, so it does not separate a trained encoder from an untrained one[^sig].
On the supplied clips neither part appears: the per-patch code forms by block 6 with no step at points 8 → 9, while the mean-pooled
curve is early under every fold grouping and at the paper's clip count. What training changes on both stimulus sets is
the per-position code from point 6 on (supplied: V-JEPA 2 0.94–0.98 against 0.22–0.38 for the random network;
hard: 0.80–0.97 against 0.57–0.62 on seed 0 and 0.52–0.61 on seed 1), not pooled availability; the half-frame dip
and jump are training-specific too (the random network's cross-half has no dip on either seed), though its level at
point 9 is not.

**Objective axis: VideoMAE** (v1 ViT-L, `MCG-NJU/videomae-large`, pixel reconstruction; 224-px input, 1,568 tokens; the paper used the VideoMAE-v2 family)[^obj]. VideoMAE matches V-JEPA 2
on every variable. Direction: block 1 0.886 vs 0.875, peak 0.992 (point 21) vs 0.991 (22), onset 2 for both. Speed:
peak 0.996 vs 0.994, onset 1. Acceleration: peak 0.996 vs 0.992, onset 1. Nested K at each model's peak is 67 / 109 /
87 for VideoMAE vs 88 / 89 / 67 for V-JEPA 2. Steering reaches the bar with 4 / 8 / 9 probes vs 4 / 7 / 6
(`figures/fig5_objective_axis.png`). Nothing in the pooled Part 1 measures on this stimulus is specific to latent
prediction; VideoMAE was not run per patch.

### 3.2 Iterative nullspace probing

**Paper's claim.** Direction needs tens of dimensions and its curve has a sawtooth. Speed needs fewer dimensions (C.4: 21–29 against
66–136; C.11: 16–31 against 14–136) and shows no sawtooth (Fig. 23 caption). **Figures:** `figures/fig2_inlp.png` (point 9) and `figures/fig2b_dim_vs_layer.png`.

| Point | Direction: nested K [fold range] / K at Fig. 22's R² < 0.3[^fig22] / paper-protocol K | Speed: nested K / paper K | Accel.: nested K / paper K |
|---|---|---|---|
| onset (2; 1 for scalars) | 289 [197–383] / 152 / 395 | 361 [350–385] / 410 | 466 [336–509] / 554 |
| 8 | 40 [33–47] / 21 / 83 | 45 [39–53] / 45 | 48 [44–51] / 50 |
| **9 (paper layer 8)** | **37 [33–42] / 23 / 46** | **39 [36–45] / 45** | **41 [36–44] / 47** |
| peak (22 / 19 / 21) | 88 [69–97] / 37 / 94 | 89 [78–100] / 103 | 67 [61–70] / 74 |
| random-init ViT, pt 9 | 24 [15–31] | 9 [7–10] | 11 [10–12] |

K counts probes, as on the paper's Fig. 22 y-axis; the text's "dimensions" is 2K for direction (point 9: 74). Source: `p1b_{var}_{target}_meanpool_L{pt}.json`
and `p1b_*_random_L{pt}.json`.

- **Random-removal band.** Projecting out a random subspace of matched rank (10 seeds) leaves the score unchanged.
  At point 9, removing 92 random dimensions leaves direction CV R² at 0.980, the same as with nothing removed. At
  point 22, removing 194 leaves it at 0.990. "Tens of probes" is therefore a real count, far outside the band.
- **Direction vs speed.** The two counts come from different clip sets (the supplied direction set: 64 θ, half
  accelerating, starts in [−2, 2]², speeds to 7; the speed set: 64 θ, constant velocity, starts in [−1.2, 1.2]²,
  0.25–4 m/s), where the paper reads both variables off one velocity set; on the same clips (the direction set's 750 constant-velocity clips, 596 train) direction vs speed K is 28 vs 22, 17 vs 26 and 59 vs 46 at points 8 / 9 / 22 (nested) and 36 vs 34, 36 vs 28 and 75 vs 45 (paper protocol), 54 vs 43 and 63 vs 48 at speed's peak 19, so under the C.11 rule speed needs fewer dimensions (2K vs K) in all 8 cells and fewer probes in 7 of 8 (nested point 9 is the exception, 17 vs 26), while under Fig. 22's thresholds it needs more probes in all 8 cells (direction 11–35, speed 23–52; the four paper-protocol speed values are floors, since the MAE rule stops the sequence at R² 0.12–0.18, before 0.1) and, nested, fewer dimensions at points 8 and 19 but not at 9 (22 vs 29) or 22 (52 vs 52); under the paper protocol the dimension comparison is undetermined for the same reason. Under the paper protocol speed's C.11 count also stops on the MAE rule, at R² 0.12–0.18, so 4 of the 7 "fewer" cells are taken at a looser point than direction's[^sameclip]. Across the two supplied sets (the table above) the probe counts are equal (37 vs 39 at point 9, 88 vs 89 at the peaks) and speed needs fewer
  *dimensions* only because its probes are 1-output. So under ridge the paper's second claim reproduces in probe count on the same clips under its method-text thresholds, reverses under its figure caption's, and across the supplied sets is a tie at point 9 and the peaks with speed fewer only at point 8 under the paper protocol: the count is set by the stop rule. The paper's plotted Fig. 22 reads about 44 direction probes and 28 speed probes at layer 8. For speed the plotted data follow C.11: Fig. 23's speed curve ends near R² 0.05 at about probe 28. For direction they do not: Fig. 23's layer-8 direction sequence runs to about 100 probes with within-15° accuracy still 20–60% near probe 44, so C.11's stop had not fired there and Fig. 22's 44 came from an earlier cutoff (an earlier cutoff is one reading: under the caption's R² < 0.3 our stored-set direction count at point 9, the paper's layer 8, is 25 under the paper protocol and 23 nested, about half of 44, and C.12 l.1244 itself reports 25 probes under R² < 0.1 at that layer), while our literal Adam C.11 direction count on the same clips, 103 at point 9, sits where Fig. 23 ends. Read that way, the caption's rule for direction and C.11's for speed, the same-clip ridge counts give speed more probes (11 vs 26 nested and 19 vs 28 at point 9; 14 vs 22 and 21 vs 34 at point 8), so the figure does not settle the call either way. The literal Adam recipe on the stored sets flips the probe-count call (Adam bullet below), but run on the same clips (batch 64, paper protocol) it gives ridge's pattern: under C.11 speed needs fewer probes at every point run, 110 vs 131, 81 vs 103, 92 vs 140 and 68 vs 198 at points 8, 9, 19 and 22 (full batch at 8 and 9: 110 vs 136, 82 vs 106), and under the caption's thresholds more, 124 vs 62, ≥ 81 vs 60, ≥ 94 vs 70 and 73 vs 52 (the middle two floors). As under ridge, every Adam speed count under C.11 is set by the MAE clause, which fires before R² reaches 0.05 in all four cells, so the "fewer" cells are again taken at a looser point than direction's R² < 0.1. Against the stored-set Adam counts (1,200 / 1,228 train clips) the same-clip direction count on 596 clips is 1.2–1.5× larger while speed's is 0.85–1.06× (batch 64; full batch 1.10–1.36 and 0.90–1.08), so the move from stored sets to same clips is carried by direction rising, not speed falling[^adamsc]. The cross-set numbers that follow are kept for the coordinate and stop-rule comparisons.
  Early layers hold each variable in hundreds of weak redundant directions (onset rows), which fits the
  random-feature picture from step 1. Along depth the direction count matches the paper's §7.2 profile ("roughly 40–50 features, increasing to up to 80 near the output layers"): nested 37–40 at points 8–11 rising to 88 at points 22–25 (paper protocol 43–47 at points 9–11, 85–94 at 22–25). Speed does not match C.11's flat "16–31": nested 39 at point 9 and 89 at point 19, above 31 at every block, so the paper's contrast of a growing direction subspace against a flat speed one does not hold here, and at points 1–2 both counts are in the hundreds where Fig. 22 shows about 2 for direction and about 25 for speed (direction 216–289, speed 361–380; the order, speed above direction, is the figure's). The counts depend on the coordinates: C.11 states no normalisation, and in raw
  centred coordinates (α re-chosen) nested K is 1.4–1.6× larger. Direction vs speed is then 51 vs 55 at point 9 and 65
  vs 73 at point 8, and under the paper-protocol rule 63 vs 61 and 68 vs 83[^raw]. So equal counts hold at point 9 and
  weaken at point 8, where speed needs more probes, not fewer.
- **The stop rule is not the same for both variables.** C.11 stops when either the R² rule or the MAE rule fires. For
  speed and acceleration the MAE rule (MAE > 0.9× the mean predictor's) fires first, at round R² 0.13–0.20 at every
  V-JEPA point under both protocols, while direction runs on to R² just under 0.1 (0.094 at the lowest). The scalar counts are therefore taken
  at a looser point than direction's. The same asymmetry means the speed and acceleration counts at Fig. 22's R² < 0.1
  (`K_loose`) are floors at every V-JEPA point except nested acceleration at onset, as in the VideoMAE
  files (footnote [^fig22]). Rerun with one R² rule for all variables[^stop], which variable needs more probes
  depends on the rule and the protocol. Nested, the scalars need more: direction / speed / acceleration 37 / 47 / 48 at
  point 9 and 40 / 60 / 61 at point 8 at R² < 0.1, and 45 / 55 / 62 and 52 / 83 / 76 at R² < 0.05. Under the paper
  protocol direction and speed are about equal at point 9 (46 vs 50; 58 vs 59), and at point 8 direction needs more
  (83 vs 57; 113 vs 74), the one cell where speed needs fewer. On the cross-set counts the paper's claim holds only in that cell; on the same clips (previous bullet) it holds in 7 of 8 C.11 cells and no Fig. 22 cell.
- **One column per round.** C.11 says "project out the learned direction", while each direction probe has two output
  columns. Removing one column per round (alternating sin/cos, or the top singular vector) takes 72–73 rounds nested
  and 84–92 under the paper protocol at point 9, about twice the stored 37 / 46, so the removed dimension count is
  about the same. Under ridge it creates no sawtooth (R² drop autocorrelation 0.82–0.84 nested, no isolated dips);
  under Adam the drops are negatively autocorrelated on acc15 (two-column full-batch R²: +0.22), and one-column removal gives direction 1–2
  isolated dips (alternating: 1 on R²; top singular vector: 2 on acc15; two-column: 0) against speed's 2, so still no
  direction-specific sawtooth[^onecol].
- **Sawtooth.** Under ridge there is none. There are no isolated dips at any direction layer under either protocol.
  The lag-1 autocorrelation of per-round drops in R² is positive everywhere (0.42–0.95; a sawtooth gives negative
  values). On within-15° accuracy it is positive under the nested protocol (0.26–0.78) and mixed under the paper
  protocol (−0.24 at point 2, −0.06 at point 8, 0.27 at point 9, 0.19 at point 22). The paper's Fig. 23 teeth are about
  65 points deep by eye; here successive probes' readouts are 9–15° apart under ridge (mean consecutive readout angle,
  points 2–22), where a sin/cos pairing would put them near 90°[^saw].
- **Sawtooth on metrics both variables share** (8-bin accuracy and R², points 8 and 9)[^saw]. On 8-bin accuracy (bin edges at label midpoints, so no label sits on an edge; an edge-on-label binning is
  kept beside it) the drop autocorrelation under nested ridge is positive for both variables (direction 0.54 / 0.55,
  speed 0.66 / 0.73 at points 8 / 9); under the Adam recipe it is negative for both (direction −0.30 to −0.39, speed
  −0.38 to −0.47), and Adam's isolated dips number 0–3 per run for speed against 0–1 for direction (one, at point 9,
  batch 64). Under ridge no direction run has an 8-bin dip. On a common metric speed is at least as jagged as
  direction.
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
MLP keeps 0.37–0.82 (direction) and 0.76–0.91 (speed) at points 9 / 12 / 22, and every planted ring drops below −0.23. In the centroid DFT, k = 2 holds 0.21–0.37 of the
non-constant power from point 8 on (0.08–0.16 at points 2–7) and k ≥ 3 together 0.03–0.11. The controls calibrate each
column: shear inflates literal K (8) but not whitened K, a planted k = 3 harmonic shows up in the DFT (0.32) with
literal K = 1, and copies do not inflate K. So (i) direction is organised on a harmonic basis (k = 1 plus a cos 2θ
term shared by opposite directions), not on tens of independent axes, and (ii) the literal count behaves like Jin et
al.'s sheared circle. "Tens of dimensions" is a conditioning count of an anisotropic rank-2 linear code, not an
intrinsic rank; beyond that code lies a nonlinear residual that grows toward the output.

**Verdict.** The "tens of dimensions" claim reproduces at the paper's layer against a random band. The claim that
speed needs fewer probes is a stop-rule and recipe call: under ridge on the same clips speed needs fewer probes under C.11's
thresholds in 7 of 8 cells and more under the Fig. 22 caption's in all 8, and the paper's plotted Fig. 22 sits with C.11's rule for speed but not clearly for direction (its Fig. 23 layer-8
direction run reaches about 100 probes); across the two supplied sets the probe
counts are equal at point 9 and the peaks (speed fewer only at point 8 under the paper protocol), under the literal Adam
recipe speed needs at least as many probes in 3 of 4 stored-set cells but fewer under C.11 on the same clips at points 8, 9, 19 and 22, and in dimensions (2K for direction,
K for speed, C.11's unit) speed needs fewer under C.11's thresholds in every cell but under the Fig. 22 caption's only
in the paper-protocol point-8 cell (nested 42 vs 60 at point 8, 46 vs 47 at point 9, 74 vs 102 at the peaks; paper protocol 84 vs 57, 50 vs 50, 78 vs
118). Direction's sawtooth does not appear under ridge. Under Adam both curves are
jagged, speed's as much as direction's on a metric both share, so the jaggedness tracks the recipe, not the variable.
The absolute counts depend on the coordinates (nested K 1.4–1.6× larger raw; paper protocol 0.8–1.8×) and on which stop rule fires, which is looser for
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
  and 0.18 at N = K at point 9, with a hump of 0.33–0.49 m/s at N = 8–12. At point 22 it is 2.85 m/s at N = K = 88. Steering speed and reading direction: 2.9°
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
  target (stored: 5); with 200 draws at every N[^n200] the rank-2K null is beaten at p < 0.05 from N = 4 for both
  (stored probe: N = 9; split-half probe: N = 11, sustained from 15), and the rank-matched null from N = 3 (stored: 5;
  split-half: 5). The
  null reading therefore depends on how the evaluation probe is regularised, which C.12 does not fix.
- **The paper's steering basis** (point 9, direction)[^adamb]. C.12 steers along C.11's Adam probe sequence, not a
  ridge one. Refitting that sequence (lr 1e-3, wd 1e-4, batch 64; it reproduces the stored Adam run and stops at K = 84)
  and steering with it: N = 1 / 2 / 3 / 5 / 10 / 20 / 84 give 84.1° / 81.1° / 77.7° / 69.2° / 27.9° / 7.3° / 2.9° to
  target (ridge basis: 78.4° / 61.1° / 25.4° / 8.7° / 3.1° / 2.9° / 2.7° at N = 37). Eighteen Adam probes reach 10°,
  against five ridge probes and the paper's about 20; N = 1–5 give 84–69°, the paper's "modest improvement (MAE > 50)".
  With 20 draws the Adam basis beats the rank-matched null at the 1/21 floor from N = 12 but falls back to p = 0.095 at
  several N up to 23; with 200 draws[^n200] it beats that null at p < 0.05 from N = 14 (sustained from 16; p < 0.01
  from 22, sustained from 24) and the rank-2K (168-d) null only from N = 30 (p < 0.01 from 75), while the ridge basis
  beats them from N = 5 and N = 9 (p < 0.01 from 7 and 18, sustained from 24). At N = 18–20 the Adam basis is at
  p = 0.02 (0.0498 at N = 14) against the same-rank null and 0.29–0.32 against the rank-2K one. The per-probe train R² of the Adam
  sequence never falls below 0.1 through round 84 (lowest 0.126), so a train-R² reading of C.12's stop would give K > 84. Her basis has K = 25, so her 20 probes are 80% of it;
  our refit has K = 84 (batch 64, our choice), so 18 is 21%; 16 reach her 12° threshold; her full basis reaches < 0.5°
  (§7.2), ours 2.9°.
- **An edit built from one probe, weighted by the covariance**[^r2cov]. The N = 1 edit above is Euclidean and stays in
  the probe's 2-D plane. The covariance-weighted edit x + ΣW(WᵀΣW)⁻¹(y* − ŷ), with W the first ridge probe (sample train
  covariance, no shrinkage; Ledoit-Wolf gives the same), lies in span(ΣW), not in the probe's plane, and reaches 3.20° to target (85.96° to true) under the stored evaluation probe, and 4.05° /
  3.61° / 5.19° under the α = 1e-3, Adam and split-half probes, against 78.4° / 58.0° / 68.6° / 82.3° for the Euclidean
  edit. A random 2-D subspace weighted the same way reaches a mean 9.6–12.0° (p = 0.14–0.33; the means carry one
  outlier draw of norm 982, the medians are 4.2 / 4.5 / 5.7 / 6.4°), with edits larger (median 2.5×, mean 5.6×) and more
  off-target speed change (median 125×, mean 390×); the learned probe beats all 20 draws on both (p = 1/21). Built from the Adam
  sequence's first probe instead (cosine 0.8 to the ridge one), the edit reaches 4.3° / 4.7° / 4.4° / 6.3° under the four
  evaluation probes against null medians of 5.0–7.0° (p = 0.095–0.24), the same picture on target error; its off-target
  speed change is 0.150 m/s against 0.033 for the ridge probe (null p = 0.095 rather than 1/21), so the specificity
  edge is the ridge probe's. The paper's unit circle in §7.1
  is a population of MLP units at fc1/fc2; this test is at the block-output residual stream, and the paper's §7.2
  sentence that steering "along a single feature direction or probe axis produces little to no change" holds here for
  the along-axis edit. The oracle (W = the evaluation probe) gives 0° by construction.

**Verdict.** Fig. 24's shape reproduces, and so does its count once the basis is the paper's: 18 Adam probes to 10°
against its about 20, where a ridge basis needs 3–5 (3–4 under the near-unregularised and Adam evaluation probes, 5
under my CV-chosen one; C.12 gives no recipe). The ridge probes beat a random subspace of their own rank from N = 2–7
(N = 3 under the near-unregularised probes) and a rank-2K random basis from N = 4 under those probes (200 draws) but only from
N = 9 under mine with 200 draws; the Adam basis beats a same-rank random basis from N = 14 (sustained from 16) and
the rank-2K basis only from N = 30, so at its N = 18–20 it beats a random subspace of its own rank but not one of the
full rank 2K. "One probe fails" is the Euclidean edit: covariance-weighted, an edit built from one probe
steers to 3–5° outside the probe's plane, though no better on target error than the same construction on a random 2-D
subspace. An untrained network
shows the same curve.

### 3.4 C.12's 70/30 split

Part 1 was rerun at C.12's 70/30 split (C.11's own split is the 80/20 used everywhere else) (`splits/split_paper70.json`: same seed, stratification and
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

Recipe (Goodfire A.3, with the PCA fit on our train clips rather than all prompts): PCA-64, one centroid per value, a periodic cubic spline for direction and a natural
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
  12, and 6.9 / 4.3 / 3.75 at point 22[^interp]; on the knot order of the label-free angle, which is not monotone in θ,
  the interpolating error is 188 at point 12 and 71.8 at point 22 (and the smoothing spline's 41.0 against the chord's
  4.46 at 22). Steered on the labels angle, the interpolating
  spline keeps the path result (radius gap +0.28 on the headline arc at point 12, +0.30 at point 22), but its endpoint
  is worse at point 12 (+2.30° on the headline arc; +2.45° ± 2.14 over 8 arcs, 3 of 8 "negative_endpoint") and mixed at
  point 22 (−0.48° on the headline arc; +1.18° ± 1.57, 2 of 8), with an edit 1.4–1.6× the chord's on the headline
  arc at each point. All steering uses the count-weighted smoothing spline, which I chose on train folds, because the interpolating
  one rebuilds held-out centroids worse and makes a larger edit.
- **Circle, ellipse or bent line?** (`p2_ellipse_direction.json`, `figures/fig4g_ellipse_direction.png`). An ellipse.
  In the plane of the ring's own cos θ / sin θ component the axis ratio b/a is 0.74 / 0.87 / 0.74 at points 8 / 12 / 22
  by a direct conic fit, 0.74 / 0.82 / 0.73 from the full-space rank-2 chart, and 0.76 / 0.76 / 0.66 from the 2θ
  distortion of the label-free atan2 angle. Points 2–4 are much flatter (0.38–0.43) and point 10 is nearly round
  (0.89). Geometric residual in centroid-noise units at points 8 / 12 / 22: ellipse 0.92 / 1.15 / 1.11, circle 1.22 /
  1.55 / 1.36, smoothing spline 0.92–0.97 (about 1 by construction). So the ellipse sits at the noise floor and the
  circle does not. The ring is also bent out of its plane: a third centroid axis follows cos 2θ (a saddle) and holds
  0.28 / 0.20 / 0.24 of the centroid variance at 8 / 12 / 22, growing to 0.34 at point 14 (the forecast carries it but
  does not read it as direction, speed or position; §4.5, saddle axis). At points 14–20 and 24 the
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
- **Ring or distorted loop? The harmonics** (Kantamneni & Tegmark's FFT over values, arXiv 2502.00873;
  `figures/fig_fft_harmonics.png`)[^fft]. At every point: PCA-64 on train clips, the 64 direction centroids, an FFT
  of each coordinate along θ, power summed over coordinates and given as a share of the non-constant power. A ring is
  all k = 1; an ellipse or a saddle fold adds k = 2; a dent or a kink adds k ≥ 3. Controls: the random-init encoder
  and a label-shuffle floor (centroid noise only). At point 12 the shares of k = 1 / 2 / 3 / ≥ 4 are
  0.70 / 0.19 / 0.010 / 0.10, at point 22 0.63 / 0.22 / 0.008 / 0.14. k = 2 is 0.06–0.11 at points 2–7 and 0.19–0.32
  from point 8 on (largest at point 15); k = 3 is at most 0.012 at every point from 1 on. The k ≥ 4 share is no more
  than noise alone gives: shuffled-label centroids carry 0.14 of the real power at point 12 (0.16 at point 22), 91% of
  it at k ≥ 4. The random-init encoder has the fundamental (0.48–0.68) and no second harmonic (k = 2 ≈ 0.03 at every
  point from 1 on). So the loop is a ring plus one k = 2 term, the ellipse and the saddle fold of the bullet above,
  and nothing at k = 3: a distorted ring of one specific, symmetric kind, and the k = 2 term is what training adds (the
  same picture as the §3.2 DFT). Two cautions. The clip bootstrap's percentile interval excludes the point estimate
  for k = 1 at every point (point 12: [0.63, 0.66] against 0.70), because resampled centroids are noisier and their
  extra power lands at high k, so the intervals measure noise sensitivity, not uncertainty in the share. And the test
  cannot call a scalar straight: an exactly straight, evenly sampled line is a sawtooth under a periodic FFT (k = 1 /
  2 / 3 = 0.61 / 0.15 / 0.07), and speed at point 12 reads 0.52 / 0.13 / 0.10.
- **Curvature vs noise.** Over any knot gap up to 45° the chord and the arc differ by less than a quarter of centroid
  noise (the contiguous design's own knot gap is 50.6°, where the sagitta reaches 0.45 of noise). Held-out centroids on the contiguous 45° arc are rebuilt best by the chord at every direction layer (point
  12: chord 2.59, smoothing spline 2.76).
- **Planted-ring positive control** (point 12). A synthetic ring is recovered without labels (angle; the stored `recovered` flag also demands a cubic win) once its radius is ≥ 0.40
  of the real ring's (circular correlation 0.996). A cubic-over-line gain appears only at 0.80[^planted]. A
  "curvature below noise" result is therefore a real null for this pipeline, not blindness.
- **Cone check / velocity plane.** On the speed set the ring's radius grows with speed and then saturates: at point
  12 it is 4.12 at 0.46 m/s and 7.99 at 3.79 m/s (ratio 1.94 for an 8.29× speed ratio; correlation 0.81), and 7.2–8.1
  from 1.4 m/s upward. Procrustes fits of the (direction × speed) cell centroids favour the ring over the velocity
  plane at points 12 and 22, narrowly, and tie at point 8 (point 12: 0.445 vs 0.428; point 22: 0.435 vs 0.421; point 8: 0.285 vs 0.285). The
  sharp test is to take a chord between opposite directions: a velocity plane predicts the speed readout at the
  midpoint collapses (ratio cos 90° = 0), and a ring predicts it is unchanged. Measured MLP-speed ratios at Δθ = 180°
  are 0.989 / 0.993 / 1.071 at points 8 / 12 / 22, and Eq. 9 speed ratios (speed × direction cell centroids) 0.991 / 0.963 / 0.953. **Verdict: ring,
  with a radius that saturates in speed, not a velocity plane**[^vp] (`figures/fig4_ring_radius_vs_speed.png`). Scaling
  the radius at point 22 does not move the forecast's speed (§4.5, radial steering).
- **Is the ring occupied?** (780 held-out clips, chart plane, ring radius 1)[^p2b]. Yes, along its whole length: the
  largest angular gap between clips is 2.9° and neighbouring directions overlap (spread 2.2× the spacing at point 12,
  4.4× at point 22). The interior is nearly empty (3.6% of 780 held-out clips inside radius 0.5) except for the slowest speed-set clips (0.25–0.67 m/s: median radius 0.59,
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
- **Two routes, held out** (`figures/fig_two_route_heldout.png`)[^trh]. The test above builds its spline on all 64
  values. Here it runs on the 16 held-out arcs of §4.3: PCA-64, centroids and spline are fit without the arc, the
  readouts on disjoint probe-fold clips, carriers are test clips, and the mean is over arcs with a bootstrap CI over
  arcs. The target h is a held-out value and the source its antipode h − 180°, so the two spline routes enter the held
  arc from opposite sides. On the probe readout each route puts 0.95–0.97 of its interior waypoints' mass on its own
  half-ring (point 12: via +90° 0.96 [0.95, 0.97], via −90° 0.95 [0.94, 0.97]; point 22: 0.96 / 0.97), in order
  (Spearman 0.98–0.99 at point 12, 1.00 at point 22), and keeps a minimum readout radius of 0.73 / 0.77 from a start of
  1.00 (point 22: 0.79 / 0.81). The chord's radius falls to 0.08 [0.08, 0.09] (point 22: 0.07), and each of its halves
  carries 0.32–0.45 of that mass. The independent MLP agrees (0.84–0.87 for the spline routes). The endpoint does not
  favour the spline: 10.7° [6.3, 17.8] against 8.5° for the smoothed-knot chord and 6.6° for the raw chord at point 12
  (spline − raw chord +4.1° [0.1, 10.5]), and 6.3° against 6.2° and 4.0° at point 22 (+2.4° [1.2, 3.7]). With the held
  arc between two kept endpoints instead (source h − 90°, target h + 90°), the route that crosses the held arc reads
  the held value at its midpoint to 15.2° [12.6, 18.3] at point 12 and 8.6° [6.9, 10.4] at point 22, against 10.7° and
  6.9° for the route over kept values (a difference of arc means, about 4.5° and 1.7°, not a paired interval). So the
  long-way-round result survives holding out the target: either spline route walks its own half of the ring into an
  unseen value, the chord collapses through the centre, and crossing unseen values costs a few degrees at the
  midpoint. The endpoint loss to the raw chord (§4.3) carries over. Point 12 runs on the label-free angle, including
  the 8 misaimed arcs of §4.3; point 22 on the labels.
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
   spline, probe clips (folds 3–4, 480 clips) fit the evaluation probe, the MLP and the Eq. 10 reference, and test
   clips are steered (48 per target, 384 steers). Beside the probe there are two readouts that did not build the edit:
   agreement R with real-clip centroids at the target value (`nearest_real_R`), and an MLP on disjoint clips (§4.4).
3. **The edit looks right only at layer L.** This leak needs the propagation and predictor readouts (§4.5).
   At the steered layer, Goodfire's Eq. 10 "behaviour" (B.1; its §5 Eq. 9 uses bin centroids) is a softmax over distances to the spline, so it restates the
   activation geometry. Every figure that uses it prints that caveat.

"Held-out" is used in three senses, and each result says which it meets. *Excluded from fitting* holds for all
Part 2 results. *Excluded from development decisions*: the spline type, k, the number of waypoints and the layers were
chosen on train folds. *Untouched*: test is read once for the table below, and anything chosen after that read is
labelled exploratory. Points 12 and 22 were chosen on train geometry; point 12 is labelled "exploratory" in the JSON
because it is neither the onset, the paper's layer nor the peak.

Design of the arms. Both arms edit the same PCA-64 subspace and add back each clip's identical off-subspace residual
(matched support). The line arm of record joins the spline's own knots, which under the smoothing spline are the
smoothed knots, not the raw centroids; the paper's A.9 comparison baseline, the chord between the raw centroids in the
PCA-64 subspace, is run as a third arm (`linear_raw`) with the same additive, residual-kept edit and reported beside
it, and at a held-out target it aims at the chord point between the raw centroids that neighbour the target's
coordinate in the knot order. That order is the label order at point 22 (labels angle) and on the headline arc, but on
8 of the 16 point-12 arcs the label-free angle is not monotone across the held-out block, so the target coordinate falls
among knots of other labels and every arm, spline and both chords alike, is aimed off the chord between the value
neighbours by up to 2.3–3.8 PCA units (arc paragraph in §4.3) (A.6's steering
baseline, which replaces the whole activation, is the separate Goodfire linear row). Exact 180° steps take whichever
sign floating-point rounding gives, where the authors' code always takes −π; endpoints are identical and only path
metrics on about one row in 63 differ. Its edit is about 10% larger than
the spline's (‖Δ‖ 8.26 vs 7.46 at point 12, 11.61 vs 10.56 at point 22) and is not dose-matched. The dose-matched
line, the ring-occupancy, 5-NN, cosine-tangent and donor-ceiling side analyses, and the BF16 and rescue counts below
still use the smoothed-knot chord. Goodfire's own linear baseline
erases the residual, so it is run separately and labelled. The
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
once it was seen to reward residual erasure (the replace arms are on-curve by construction, so Eq. 10 favours them). The
final rule is symmetric: a gain or a loss counts only beyond the same practical margin, and an endpoint loss beyond that
margin overrides path gains ("negative_endpoint"). It was frozen at commit 8d3cac8, before the multi-arc sweep and the
70/30 reruns. The frozen rule was applied unchanged to all 32 arc runs and to every steering file cited here.

### 4.3 Results

**Direction, contiguous 45° arc** (manifold = spline, linear = chord; gaps are manifold − linear with a 95% paired
clip bootstrap):

| | pt 12 spline | pt 12 line | gap [CI] | pt 22 spline | pt 22 line | gap [CI] |
|---|---:|---:|---|---:|---:|---|
| endpoint probe error (line = smoothed-knot chord) | 9.73° | 9.62° | +0.11 [0.05, 0.18] | 10.69° | 6.89° | +3.80 [3.36, 4.23] |
| endpoint probe error, line = raw-centroid chord (A.9)[^rawchord] | 9.73° | 4.71° | +5.02 [3.94, 6.06] | 10.69° | 3.64° | +7.05 [6.03, 8.08] |
| nearest-real agreement R (line = smoothed-knot chord) | 0.196 | 0.197 | −0.0004 [−0.0018, 0.0008] | 0.175 | 0.180 | −0.005 [−0.006, −0.003] |
| nearest-real agreement R, line = raw-centroid chord | 0.196 | 0.185 | +0.011 [0.002, 0.020] | 0.175 | 0.165 | +0.011 [−0.001, 0.023] |
| min readout radius along path | 0.86 | 0.61 | +0.26 [0.22, 0.29] | 0.85 | 0.61 | +0.24 [0.21, 0.27] |
| A.7 energy on the Eq. 10 behaviour ÷ real-clip floor | 0.84 | 1.42 | | 1.01 | 1.00 | |
| intermediate mass on the arc | 0.68 | 0.48 | | 0.65 | 0.45 | |
| waypoint ordering (Spearman) | 0.90 | 0.79 | | 0.76 | 0.67 | |
| reflected arm: radius / energy / ordering | 0.53 / 1.64 / 0.52 | | | 0.51 / 1.12 / 0.50 | | |

Source: `p2_steer_direction_direction_L{12,22}_contiguous.json`; the raw-chord row and the sagittas come from the
rerun `p2_steer_direction_direction_L{12,22}_contiguous_rawchord.json`, whose old-arm values reproduce the stored
file to 1e-9 relative. The smoothed-knot chord sits 0.3–1.1 (point 12) and 0.8–2.1 (point 22) PCA units from the
spline at the eight held-out targets; the raw-centroid chord sits 1.9–2.4 and 3.2–3.7, so the paper's baseline is
further from the spline than our line arm was, and lands closer to the target. Against the raw chord the spline's
agreement with real clips at the target is higher on the headline arc at point 12 (+0.011), but over the 16 point-12 arcs the gap is −0.021 ± 0.061 SD (spline ahead on 7, CIs split 4 above and 4 below zero), while at point 22 it is +0.018 ± 0.013 (ahead on 15 of 16, 11 CIs above zero; `results/arcs_rawchord/*/gaps.manifold_minus_linear_raw.nearest_real_R.mean`). So against the paper's chord the nearest-real readout favours the spline consistently at point 22 and not at point 12; the "independent readouts null" reading below is a smoothed-knot-chord result at point 22 and stands at point 12 either way. Paired per-pair SEs (A.9's form) are 0.43 and 0.41 for the raw-chord endpoint
gap; the table keeps the clip bootstrap used by every other row. Figures:
`figures/fig4_waypoint_readout_direction_direction_L12_contiguous.png` (radius and Eq. 10 distance along the path) and
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

The path result holds on every arc at both points (with the point-12 knot-order caveat below, which moves every arm's
target alike), and against the smoothed-knot chord the endpoint ties at both
(the raw-centroid chord over the arcs is in the paragraph after the donor ceiling). At point 22 the one
"negative_endpoint" run is 191.25°–230.625° (+3.7°) and the other 15 are within ±3°, so the single-arc +3.80° above is
not typical. On the label-free angle the four rerun arcs had the two largest endpoint losses (+9.2° and +9.9°, the
duplicated arc) and the four smallest radius gaps (+0.11 to +0.19); on the labels they give +0.21°, +0.27°, −0.36° and
−0.25°, and radius gaps +0.26 to +0.32.

**Donor ceiling** (point 12 / 22, same carriers). Replacing the carrier's PCA-64 coordinates with those of a real
unseen test clip at θ\* reaches nearest-real R = 0.23 / 0.20, against 0.19 / 0.17 for both spline and line. Swapping
the whole activation adds almost nothing to R (+0.004 / +0.001). A real clip at θ\* reaches only R ≈ 0.2 against the
other real clips at θ\*, so this readout's ceiling is set by clip-specific variance, and the spline reaches 84% / 84% of the in-subspace ceiling (82% / 84% of the whole-activation donor)
of it. The MLP evaluator reads the in-subspace donor at 25.2° / 18.5° vs 30.7° / 21.2° for the spline[^donor]
(`figures/fig4d_donor_ceiling_direction_L12_L22.png`).

**Raw-centroid chord over the same 16 arcs** (the `linear_raw` arm rerun on every arc; point 22 on the all-labels set;
`results/arcs_rawchord/L{12,22}_s{1..16}/` and `L22_s{4,8,9,10}_labels/`, `gaps.manifold_minus_linear_raw.probe_err_to_target.mean`,
the field the table above uses)[^rawchord]. The paired endpoint gap (spline − raw chord) is +1.32° ± 1.77 SD at point 12
(range −1.65 to +3.97; the spline ahead on 3 of 16; 11 runs with a clip-bootstrap CI excluding zero, 10 above and seed 1 below (−1.65 [−2.43, −0.94]); +1.14 ± 1.68 with
the duplicate arc counted once) and +2.33° ± 2.01 at point 22 (range +0.38 to +8.28; the spline ahead on none; 15 CIs
excluding zero; +2.45 ± 2.02 counted once), against +0.08 ± 1.47 and +0.17 ± 1.33 for the smoothed-knot chord on the
same runs. So the endpoint tie in the table above is a property of our smoothed-knot line arm; against the paper's
chord the spline loses the endpoint on average at both points, modestly, and the headline arc's +5.0° is the
largest of the 17 runs at point 12, and its +7.0° is second at point 22 to seed 16 (+8.28°). Radius, ordering and A.7 energy do not change with the choice of chord; the raw
chord sits further off the reference curve and further from real clips than the smoothed chord did, and its edit is
about 10% larger. The rerun reproduces the stored arms' summary means to 5e-10 relative on every arc but one (L12 seed 9, 1.3e-5).
One caveat found by the last audit: on 8 of the 16 point-12 arcs (seeds 1, 4, 5, 7, 8, 9, 11, 13) the label-free
centroid-plane angle is not monotone across the held-out block, so `coord_of_value` places the held-out target among
knots of other labels (seed 4: the 298.125° target lands between the 315° and 309° centroids) and every arm is aimed off
the chord between the true value neighbours by 2.3–3.8 PCA units. All arms' endpoint errors are larger there (spline
8.7° vs 7.3°, raw chord 7.8° vs 5.5°). On the 8 clean arcs (seeds 2, 3, 6, 10, 12, 14, 15, 16) the raw-chord gap is
+1.77° ± 1.32 (1 of 8 favours the spline) and on the 8 affected arcs +0.87° ± 2.12 (2 of 8); the smoothed-knot gaps are
−0.19 ± 1.22 and +0.35 ± 1.73. The headline arc and every point-22 arc (labels angle, monotone) are clean, so the
verdict does not move, and the misaimed arcs were not rerun. The validator's later `aim="arc"` option (commit 42b30fe)
fixes only the chord's target on these arcs; the spline's knot-a-to-knot-b stretch still passes knots of other values
there (3 on seed 4, 1 on seed 1), so the spline side of the misaim is open, not fixed.

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

Cross-render transfer (`p2_steer_direction_direction_L{12,22}_contiguous_ctx-hard.json`): the same splines and chords
steer 48 held-out clips per target from the textured hard render, read by a probe fit on hard-render probe folds (the
supplied-render probe reads unsteered hard clips 41.5° / 79.3° off, so it cannot score these edits). Spline minus
matched chord at the endpoint: −0.03° [−0.42, 0.32] at point 12, +3.83° [2.82, 4.77] at point 22 (chord ahead). The
point-22 call is on a knife-edge: the spline's endpoint loss over pairs (3.90 ± 0.59°) sits just under the verdict margin
when that margin is the probe's mean error on unsteered hard clips (4.03°, call "path-geometry positive") and just over
it under the median (3.57°, call "negative: spline worse at held-out endpoint"). Both floors are on-grid estimates:
unsteered hard clips exist only at the render's 8 angles, while 7 of the 8 held-out targets lie between them
(`verdict.margin_floors`, `verdict.margin_note`).

Position sheet (speed set, start (x, y), thin-plate spline vs chord to a held-out interior 2×2 block): at point 12
"negative: TPS path indistinguishable from chord". At point 19 "chord better than the TPS path on err_path and
excess_to_nearest_real". Endpoint error is 0.178 m for both vs 0.197 for a Delaunay interpolation (point 12)[^sheet].

**Reading.** At the encoder layer the spline stays on the ring (the Eq. 10 distribution walks the arc in order) and the
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
through the ring" holds in the ring plane only among the unwhitened spaces (whitened direction subspaces: next paragraph), and so does Goodfire's activation-side density premise (its §3.4: e^{−E} "small where they are sparse (off M_h)") here. For speed and
acceleration all arms coincide with the chord inside the knots; in extrapolation, continued along its end tangent, the
smoothing spline trails the chord by 0.02–0.06 with the CI clear of zero on all four (all "path_geometry_positive"), and the authors'-code arm beats its
chord on speed and trails it on acceleration. Extending the end cubic piece past the last knot instead, my choice and
not the method's, gives large extrapolation losses.

**Local density where direction is read** (`figures/fig_local_density.png`)[^dens]. The paragraph above finds the
hollow in the ring plane and not in unwhitened 64-D. Is that because unwhitened distances are dominated by variance
that a direction reader ignores? Design (a density readout in the spirit of Goodfire's Eq. 6, which the paper defines
but does not measure; spline and smoothed-knot chord on all 64 values from knot clips, so a question about where
midpoints land, not a held-out steering test): for every test clip and shifts of 90°, 135° and 180° in both senses,
each midpoint's mean distance to its 5 nearest train clips is divided by the same distance for real test clips at the
midpoint angle (1 = as dense as real clips there), in five spaces: full, PCA-64, a whitened (sin, cos) plane
(LEACE-style: the plane any linear direction reader uses, with every direction of the data at unit variance), a
within-value-whitened 8-D discriminant subspace, and the chart plane. At 180° at point 12 the chord midpoint sits at
8.89× the real clips' 5-NN distance in the whitened plane against 0.92× for the spline (paired log-ratio gap +2.27
[2.23, 2.31]; 100% of chord midpoints and 4.5% of spline midpoints beyond the real clips' 95th percentile), and at
2.57× against 1.17× in the 8-D discriminant subspace; in PCA-64 and full space the gap is small, 1.17× against 1.09×
and 1.14× against 1.08×. The gap grows with shift (whitened plane: 1.55× / 4.36× / 8.89× for the chord at 90° /
135° / 180°, the spline 0.92–1.03×) and survives a reference set disjoint from the spline's knots (probe folds only:
5.92× against 0.91×). At point 22 the whitened spaces agree (plane 10.76× against 1.40×; 8-D 3.3× against 1.70×),
while in PCA-64 and full space the sign flips (the spline midpoint sparser by 0.06 in log ratio) and the chart plane
ties (1.21× against 1.25×, CI across zero). Point 8 gives the same whitened-space result (5.41× against 1.08×), but
this run builds every spline on the label-free angle, which fails at point 8 (§4.1), so its other numbers are not
read. Local intrinsic dimension (Levina–Bickel, k = 10, held-out clips) is 12.9 in full space and 10.8 in PCA-64 at
point 12 (7.2 / 6.7 at point 22) and 1.93–2.05 in the whitened plane at all three points: in the reader's plane the
cloud is locally a filled 2-D band, not a thin curve, which fits a ring radius that grows with speed (§4.1). Reading:
once the variance a direction reader ignores is whitened away, the chord's midpoint at large shifts lies far outside
the data and the spline's lies among it, at points 12 and 22; in unwhitened space that variance swamps the
difference, consistent with the null 5-NN results above (those are held-out arcs, so the match is not exact). This is a density statement about all-value paths; the
held-out endpoint verdict does not move.

### 4.4 Controls and the comparison with Part 1

- **Random curves** (20 endpoint-matched draws, point 12 contiguous). Endpoint readouts match by construction. On the
  path, the spline ranks 1/21 on off-curve excess (0.15 vs a band of 0.68–0.88) and on Eq. 10 energy (0.92 vs 1.48–1.75).
  Unmatched random curves have endpoint error 87.9°. **BF16**: the winner on both energy metrics is unchanged.
  **Dose-matched line**: endpoint 9.41°, energy 1.40, radius 0.61, so the line's deficit is not a matter of dose.
- **Goodfire's own linear baseline** (the whole activation replaced by a chord point). Nearest-real R is 0.625 and
  endpoint error 1.52°, against 0.547 for Goodfire's manifold arm (run on our smoothing spline with the same held-out target coordinates) and 0.196 for our additive arms. Erasing the residual
  makes the activation look much more like the target centroid, so Goodfire's comparison mixes "residual erased" with
  "curved vs straight".
- **Bake-off at matched edit norm** (all arms rescaled per clip to the spline's ‖Δ‖; errors from the linear probe /
  an MLP on disjoint probe clips; unsteered 88.9° / 92.3° at point 12, 88.7° / 86.4° at point 22)[^bake]:

| Arm (nominal rank) | pt 12 probe / MLP | pt 22 probe / MLP |
|---|---|---|
| spline (64; effective 2.55) | 9.7° / 30.9° | 10.7° / 21.3° |
| chord between smoothed knots (64) | 9.4° / 32.6° | 7.4° / 19.7° |
| chord between raw centroids (64; A.9)[^rawchord] | 6.4° / 31.6° | 8.0° / 20.1° |
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
at held-out endpoints the paper's raw-centroid chord lands 5.0° / 7.0° closer than it on the headline arc, and it ties
only the chord between its own smoothed knots; the interpolating version rebuilds held-out centroids worse
than the smoothing one and edits 1.4–1.6× more than the chord (hence smoothing); on scalars it adds nothing inside the
knots and, as a smoothing spline, trails the chord by 0.02–0.06 in extrapolation even continued along its end tangent as
the authors' code does (their own interpolating arm is mixed: better on speed, worse on acceleration). **Failure cases**: the
held-out endpoint against the raw-centroid chord at both points on the headline arc (+5.0°, +7.0°) and on 13 of 16
point-12 arcs, and the position sheet.

- **Conceptor steering (COAST, arXiv 2605.17144): the subspace-steering comparison**[^coast]. A conceptor
  C = R (R + α⁻² I)⁻¹, with R the covariance of a condition's mean-centred states, is an ellipsoidal soft projection
  onto that condition's principal directions; COAST steers with C_steer = C_target AND NOT C_source, applied as a gate
  h' = h [(1 − β) I + β C_steer]. I fit it in the spline's PCA-64 coordinates, on the headline arc and its 384 steers
  (the spline and raw chord recomputed there reproduce the stored endpoints to 1e-13 relative), with target and source
  conditions made of knot clips weighted as the raw chord interpolates, the aperture by COAST's own Stage-2 rule
  (A.10.2) and each clip's off-subspace residual kept. Arms: `coast_a`, the gate on uncentred coordinates (COAST gates
  the raw hidden state), β from 0 to 1 along the path; the same stopped at β = 0.3; the gate on PCA-centred coordinates;
  `coast_b`, an update aimed at the target, z' = z + s C_steer (μ_target − z), which is my variant, not COAST's; and
  `coast_b` rescaled to the raw chord's norm. Nulls replace C_steer with a random orthogonal projector of rank
  round(tr C) (20 draws, 16 clips per target). Unsteered error 88.9° / 88.7°; readouts as in §4.3; Δ speed and Δ start
  from ridge probes on the probe folds (start position in metres, unlike the px of the next bullet):

| Arm | endpoint error, pt 12 [95% CI] | pt 22 | ‖Δ‖ ÷ raw chord, 12 / 22 | ring-plane share of ‖Δ‖², 12 / 22 | Δ speed (m/s), 12 / 22 | Δ start (m), 12 / 22 |
|---|---|---|---|---|---|---|
| spline | 9.7° [8.9, 10.6] | 10.7° [9.7, 11.7] | 0.90 / 0.91 | 0.64 / 0.60 | 0.19 / 0.26 | 0.18 / 0.20 |
| raw chord (A.9) | 4.7° [4.3, 5.2] | 3.6° [3.3, 4.1] | 1 / 1 | 0.59 / 0.60 | 0.49 / 0.49 | 0.39 / 0.40 |
| COAST gate, uncentred | 165.0° [163.8, 166.1] | 41.8° [38.9, 44.5] | 22.1 / 10.2 | 0.09 / 0.01 | 15.2 / 1.87 | 5.21 / 10.50 |
| same, stopped at β = 0.3 | 114.4° [108.4, 120.3] | 85.7° [78.6, 92.9] | 6.6 / 3.1 | 0.09 / 0.01 | 4.57 / 0.56 | 1.56 / 3.15 |
| COAST gate, centred | 142.2° [137.6, 146.9] | 127.5° [123.6, 131.7] | 1.27 / 1.15 | 0.32 / 0.31 | 1.65 / 1.47 | 1.39 / 1.32 |
| aimed (mine) | 87.8° [80.5, 95.4] | 80.2° [72.3, 88.3] | 0.11 / 0.42 | 0.12 / 0.14 | 0.20 / 0.47 | 0.07 / 0.35 |
| aimed, at the raw chord's norm | 77.5° [69.4, 85.3] | 69.7° [61.3, 77.5] | 1 / 1 | 0.12 / 0.14 | 1.69 / 1.19 | 0.62 / 0.90 |

  No conceptor arm steers direction here. The best, the uncentred gate at point 22 run out to β = 1 (a dose COAST
  itself excludes: it keeps β ∈ {0.1, 0.3}, and at its β = 0.3 the same gate sits at 85.7° against 88.7° unsteered,
  so the COAST-faithful row is a null), stops 41.8° from the target with
  an edit 10× the raw chord's, 1% of it in the ring plane, that moves the start-position readout by 10.5 m (test
  starts lie 1.51 m from their centroid on average); at point 12 the same gate overshoots to 165°, worse than
  unsteered and worse than all 20 random projectors of its rank on the same clips (165.8° against 153.7°), while at
  point 22 it beats all 20 (44.3° against 70.7°). The aimed update barely moves the readout at its own dose
  (0.11× the chord's norm at point 12) and reaches 70–78° at the chord's; it beats 19 and 20 of 20 random projectors
  (85.4° against 85.8°, 78.2° against 83.0°), so the conceptor carries a little direction information, not much.
  The selection rule already flags the mismatch: COAST keeps an aperture whose mean source–target overlap lies in
  [0.85, 0.95], and here the overlap is 0.40–0.50 at point 12 and 0.50–0.62 at point 22 over α ∈ {0.1, …, 10}, so no
  aperture qualifies and the closest is used (α = 0.1 and 0.5). COAST's pseudoinverse AND is not a valid conceptor
  (eigenvalues outside [0, 1]) for 96–100% of (kept value, target) pairs at every aperture, and for 87% / 90% of the
  steered pairs at the chosen one, so I used Jaeger's range-intersection AND (Jaeger 2014, arXiv 1403.3369; the
  singular-case formula in `src/wm/conceptor.py` was written from memory of that paper and has not been checked
  against its text); the resulting C_steer has trace 0.66 of
  64 at point 12 and 4.7 at point 22. Why it fails on a ring: a conceptor describes the shape of a condition's cloud
  after centring, not where the cloud sits, and direction is carried by where each value's centroid sits on the ring,
  while neighbouring values have clouds of nearly the same shape (§4.1: the ring is occupied along its whole length).
  C_target AND NOT C_source therefore keeps little (the trace above), and what it keeps is spread the direction
  readout ignores: 1–14% of each conceptor edit lies in the ring plane (31–32% for the centred gate) against 59–64%
  for the spline and chord. COAST's own setting, success and failure rollouts as differently shaped spreads in an
  overlapping region, is the case the AND-NOT isolates; a cyclic variable whose values are translated copies of one
  cloud is the case it cannot represent. Caveats: one arc per point (the 16-arc aggregate was not run), the aimed
  arm is mine, and COAST's final choice among in-band apertures uses rollouts that do not exist here (no aperture was
  in band anyway). This is a negative for conceptor steering of a ring code at these two points, not for COAST on
  its own task (`figures/fig_conceptor_direction_L{12,22}.png`).
- **Off-target: speed and start position** (Bao et al., arXiv 2608.23526: an edit should leave the quantities it does
  not target on their level set)[^offt]. The headline run's steered states are regenerated (edit norms match the
  stored ones to 1e-12 relative at point 12 and 1.7e-6 at point 22) and read by a speed probe fit on speed-set train
  clips and a start-position probe fit on direction-set train clips. The off-target change is |readout(edited) −
  readout(unedited)| per steer, with a clip-bootstrap CI, and in parentheses its ratio to the natural spread (the
  pooled within-label SD of the unedited readout: 0.140 / 0.169 m/s and 7.2 / 5.2 px at points 12 / 22):

| Arm | pt 12: Δ speed (m/s) | pt 12: Δ start (px) | pt 22: Δ speed (m/s) | pt 22: Δ start (px) |
|---|---|---|---|---|
| spline | 0.187 [0.173, 0.203] (1.33) | 6.9 [6.3, 7.4] (0.95) | 0.309 [0.289, 0.331] (1.83) | 6.1 [5.6, 6.6] (1.17) |
| chord between smoothed knots | 0.184 [0.169, 0.199] (1.31) | 6.2 [5.7, 6.7] (0.86) | 0.298 [0.278, 0.319] (1.76) | 4.6 [4.1, 5.0] (0.88) |
| chord between raw centroids (A.9) | 0.431 [0.397, 0.468] (3.07) | 12.1 [11.2, 13.1] (1.68) | 0.451 [0.416, 0.490] (2.67) | 12.9 [12.0, 13.8] (2.48) |
| Part 1 probe-QR at the spline's norm | 0.124 [0.112, 0.136] (0.88) | 7.3 [6.8, 7.9] (1.02) | 0.172 [0.156, 0.190] (1.02) | 20.6 [19.0, 22.1] (3.95) |
| random smooth curve (20 draws) | 0.544 [0.491, 0.594] (3.88) | 51.6 [47.5, 55.2] (7.14) | 0.687 [0.628, 0.741] (4.07) | 45.2 [41.7, 48.7] (8.68) |

  The spline and the chord between its smoothed knots leak alike into speed (1.33 against 1.31 natural spreads at
  point 12, 1.83 against 1.76 at point 22; the intervals overlap, though these are not paired tests), and into position
  alike at point 12 (0.95 against 0.86); at point 22 the spline moves position more (1.17 [1.07, 1.27] against
  0.88 [0.79, 0.96]). The paper's raw-centroid chord, which wins the held-out endpoint (§4.3), leaks more: 2.3× the
  spline's speed change at point 12 and 1.5× at point 22, with a signed shift of +0.30 / +0.31 m/s (the edited clip
  reads faster), and 1.8–2.1× its position change. So the raw chord's endpoint lead comes with a larger change in the
  variables it should leave alone; the endpoint verdict does not move, but the lead is not free. The conceptor run's
  own probes (previous bullet) give the same ordering. Part 1's probe-QR edit at the spline's norm leaks least into
  speed (0.88 / 1.02) but moves the start position by 3.95 spreads at point 22 (5.61 at its own norm). Scale: the
  transferred speed probe is itself 0.56 / 0.44 m/s off on unedited direction test clips (r 0.97 / 0.98), the same
  order as every non-random change above, so the ratios to natural spread and the between-arm contrasts, not the m/s,
  carry the comparison.

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
- **Shared vs source-specific edit.** The fraction of edit energy shared by all steered clips (n‖mean Δ‖² / Σ‖Δᵢ‖²,
  `shared_delta_fraction`) is 0.41 for the spline and 0.37 for the line at point 12, and 0.44 / 0.39 at point 22. More
  than half of the edit energy varies with the source value (the shift and transport edits depend only on source value
  and target, so no carrier-specific information enters), so spline and line are not merely one constant shift plus a
  small correction.
- **Isometry with the straight-line baseline** (Goodfire A.5, behaviour = the predictor's forecasts)[^p2b]. Pearson r is
  0.86 / 0.96 / 0.69 at points 8 / 12 / 22 with distances along the smoothing spline and 0.80 / 0.90 / 0.78 with chord
  distances, reversed at point 22 on the stored coordinate (see below). The forecast and encoder-output spaces give the same r within 0.02 as the bare angle
  difference, so on a ring this test measures whether arc length is proportional to angle change. Goodfire's world-model
  behaviour manifold is built from activations (its Eq. 10), so its 0.996 isometry has the same circularity as our Eq. 10
  figures. With the authors' recipe (behaviour manifold = an interpolating spline through the per-value forecast
  centroids in the full 1,024-d forecast space) on the label-free knot order, r is 0.84 / 0.93 / 0.67 along the
  interpolating activation spline against 0.73 / 0.87 / 0.75 for the chord, and 0.885 / 0.979 / 0.758 against 0.800 /
  0.876 / 0.752 along the smoothing spline[^iso]. On the labels coordinate (Goodfire's 70B cyclic configs and two 8B weekdays demo configs) the spline leads at every point: interpolating
  0.986 / 0.982 / 0.984 against chord 0.727 / 0.867 / 0.749 at points 8 / 12 / 22, smoothing 0.984 / 0.994 / 0.979
  against 0.858 / 0.890 / 0.831[^isol]. On Goodfire's own cyclic coordinate (the √variance-scaled atan2 its 8B weekdays
  and months runs use; its periodicity test fails at points 8 and 12 and passes at 22 by 0.005; circular correlation
  with θ −0.40 / 0.65 / 0.91) the chord leads at every point: interpolating 0.50 / 0.32 / 0.66 against 0.73 / 0.87 /
  0.75, smoothing 0.50 / 0.50 / 0.74 against 0.69 / 0.87 / 0.76[^isog]; at points 8 and 12 that run keeps the atan2 angle
  where causalab would fall back to PC1 with a natural spline, and it leaves the behaviour side on the labels where
  causalab applies the same label-free rule. Run causalab's way on both sides (PC1 fallback at points 8 and 12, its
  angle at 22; the behaviour side on its own √variance atan2, which passes its test but correlates with θ at only 0.39,
  so it caps every r near 0.3), the figures are interpolating 0.27 / -0.01 / 0.32 against chord 0.32 / 0.34 / 0.26 and
  smoothing 0.25 / 0.03 / 0.32 against 0.30 / 0.34 / 0.26[^isof]. With paired 200-draw bootstraps on geo − lin (the
  stored intervals in `p2_isometry_linear.json`, clips resampled within each value, are biased low for the chord: 20
  of 48 exclude their own point estimate; so three interval types are reported, percentile, recentred and basic), the
  calls are: labels, spline at every point under every interval but one (point 8, smoothing, basic: tie); our label-free knot order, tie at every point (spline
  at 8 and 12 under the basic interval only); Goodfire's angle, chord at 12 under every interval, chord-or-tie at 8,
  tie at 22; the fully faithful run, chord at 12, tie at 8, and at 22 tie under two intervals and spline under the
  basic one. So the isometry verdict is set by the knot coordinate, not by the curve: only the labels ordering makes
  the spline near-isometric, and under every label-free ordering the point-22 test is a tie, with the sign of any
  lead depending on the interval method. The point-22 reversal is undecidable
  without labels.

**Energy geodesic (Goodfire Eq. 4–6), lite run**[^geo]. The paper defines the geodesic as the shortest path under
G_E(h) = (α e^{−E(h)} + β)^{−1} I (l.1396–1411) but never computes one, and causalab has no implementation, so
this is ours: 50 free nodes between pinned endpoints, Simpson quadrature, torch L-BFGS, two energies fit on knot clips
only (a kNN energy with the Levina–Bickel dimension, 13.1 at point 12 and 7.3 at point 22, and a whitened top-10-PC
KDE with cross-validated bandwidth), α, β calibrated as in Béthune et al. 2505.18230 §3.3 since the paper gives no
values. Lite scope: 3 of 8 held-out targets, 16 carriers, ≤ 100 L-BFGS steps, and none of the 96 solves reached the
tolerance, so all path lengths are upper bounds; endpoints are pinned to the raw chord's end state, so endpoint error
and nearest-real R equal the chord's by construction. Result: the geodesic does not follow the ring. Its bend has an
in-plane share of 0.08–0.13 (spline 0.13–0.35, random direction 0.03) and a cosine with the spline's bend of −0.00 to
0.08 from the chord initialisation; it sits 1.8–2.7 from the chord and 3.1–4.1 from the spline. On the readouts it
matches the chord (minimum readout radius 0.77 / 0.69 for the kNN / KDE geodesic at point 12 against 0.70 chord and
0.90 spline; A.7 E_BC 1.35 / 1.18 against 1.42 and 0.86; the spline is better on both in all 3 targets, CIs excluding
0), while it wins on what it minimises (its own length 8.2 vs 14.3 chord vs 133.6 spline under the kNN energy; excess
distance to the nearest real clips −0.66 vs +0.25 vs +1.09). At point 22 the same holds (radius 0.70 / 0.68 vs 0.71
and 0.88). So under a density metric fit on the knot clips the shortest path is close to the straight one, and the
spline's ring-following is not what the energy geodesic selects; the full run (8 targets, 3 restarts, tol 1e-5) is in
progress and will replace these numbers. Point 12 uses the label-free angle and point 22 the labels, as elsewhere.

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

Edits at points 2–12 wash out: four blocks later the error is 82–88° for every structured arm but the overshooting spline (random 92°), and
at point 25 MAE-to-true is back to 4–7°. The steered-point columns are read by a probe of the family that built the
edit, so they are not independent evidence. Point-22 edits survive the last three blocks (15–25° at point 25) with
weak target specificity: against the neighbouring target (at most 39° away, 16.5° on average[^s2]) the error rises by
only 1–6° at point 25 (7–14° at the steered point). The projection on the twin's real activation change at point 25
is 0.16 / 0.23 / 0.22 (probe-QR / smoothed spline / chord) vs 0.15 / 0.21 / 0.21 for the shuffled twin (`readout_b`).
The interpolating spline on the label-free knot order overshoots (‖Δ‖ 7.2× the natural twin change at point 12, vs
0.5–0.6×) and is excluded; on the labels order its edit is 0.97× the twin change[^il12].

**Predictor** (context frames 1–8 edited at every token; the predictor forecasts tubelets 4–7). *First attempt
(blind).* Probes fit on the encoder's real future tokens (3.4°, 6.7–8.4 px) read the unedited forecast 61° / 67.9 px
off[^fpos] (62.6° / 65.7 px when recomputed on the second attempt's test clips); token-space R (`session2_predictor.json`) is 0.0185 [0.0173, 0.0195] for the twin's context, ≤ 0.006 for
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

*Twin-difference edit*[^twd]. Does an edit built from the model's own counterfactuals beat the curve arms? This is
Liu et al.'s (2608.15156) counterfactual-minus-factual carrier in simplified form, with no learned affine map. I rendered 128 knot clips (2 per source value, disjoint from the carriers and from the 480
probe clips) at every held-out target and took the point-22 activation difference, twin minus clip, over the 1,008
pairs. Each carrier's edit is the mean difference of the 4 pairs at its target whose source angle is nearest its own,
either as is ("full") or projected on the top r singular vectors of all 1,008 differences (r = 1, 2, 4, 8 hold 40%,
58%, 73% and 86% of their energy). It is then rescaled and added exactly as in the norm-matched rerun: same 200 carriers × 4
targets, probes and comparison edits, and the unscaled chord re-forwarded here matches the session-2 cache exactly. Forecast
direction error to target / R dir (unedited 92.1°):

| Edit | at the chord's norm | at the natural norm |
|---|---|---|
| twin difference, rank 1 | 59.4° / 0.29 | 43.1° / 0.48 |
| twin difference, rank 2 | 55.0° / 0.35 | 34.9° / 0.56 |
| twin difference, rank 4 | 50.2° / 0.47 | 30.4° / 0.82 |
| twin difference, rank 8 | 43.2° / 0.49 | 30.0° / 0.83 |
| twin difference, full | 41.3° / 0.51 | 28.7° / 0.86 |
| interpolating spline | 19.5° / 0.68 | 12.3° / 1.21 |
| probe-QR | 23.2° / 0.64 | 18.8° / 1.04 |
| chord | 27.2° / 0.68 | 25.7° / 1.37 |

The best twin arm (full) trails the best comparison arm, the interpolating spline, by 21.8° [17.8, 25.9] at the
chord's norm and 16.4° [13.9, 19.0] at the natural norm (paired over carriers), with R dir lower by 0.17 [0.14, 0.20]
and 0.35 [0.24, 0.46]. The comparison arm was picked post hoc as the lowest error of three, which can only favour it. The
asymmetry runs the other way too: the twin edit is built from clips rendered at the held-out target values, which no
curve arm ever sees, so this negative is conservative for the curves. The error falls with rank at both norms, so rank truncation
is not the cause. The twin edit is better on position (2.8 px [1.8, 3.7] and 3.5 px [2.4, 4.8] closer to the twin's
true disk than the spline). Read at the encoder output (the full edited clip propagated to point 25, as in the next paragraph), it ties the spline on
direction: 25.0° vs 23.9° (+1.1° [−0.02, 2.3]) and 16.1° vs 16.8° (−0.8° [−1.7, 0.2]). It also ties on nearest-real R (+0.004
[−0.009, 0.017]; −0.003 [−0.021, 0.016]), and it moves the activation furthest along the twin's own point-25 change
(R 0.31 vs 0.19, +0.12 [0.10, 0.13]; 0.48 vs 0.28, +0.20 [0.18, 0.22]). So the edit made of real counterfactual
differences is the most twin-like of these arms at the encoder output. In the forecast it is worse than the
interpolating spline, probe-QR and the chord at both norms (the smoothing spline, 39.1° / 37.3° above, was not in
this run). My reading, which I did not test directly: at a fixed norm the forecast's direction readout responds to the
direction content that the curve edits isolate, not to the whole activation change a re-render causes. A negative for this baseline, on one fit set (seed 0, 2 clips per value) at point 22 only
(`figures/fig_twin_difference.png`).

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
Paired spline − chord is +3.1° [2.1, 4.1] at own norms, +10.8° [9.4, 12.6] at the chord's norm (point 22: −7.7° [−10.6,
−5.0]) and −2.1° [−3.8, −0.4] at the natural norm (point 22: −13.4°). "The spline beats the chord at matched norm" is a
point-22 result; at the encoder output the verdict is mixed: the chord is better at its own norm and at the chord's
norm, the spline is better at the natural-change norm (2.1° and 13.3 px), and the matched-norm effect shrinks from
13.4° to 2.1°. For comparison with Goodfire: its §5 world-model evidence is about the intermediate waypoints,
decoded frames that teleport along the linear path and move smoothly along the manifold path, with the pullback of
C.3 behind it (`refs/steering_paper.txt` l.1850–1880, C.3). The point-25 run here reads the endpoint only; the
along-path readout (§4.5 above) was run at point 22, not at point 25, so at Goodfire's own site the intermediate-state
claim is untested here.

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

**The reverse test**[^ap]. For 20 carriers × 2 targets at 135–180°, I optimised 8 free waypoints (the start fixed at zero edit) in
PCA-64 at point 22, one path per carrier, added to the carrier's activation, so the forecast follows the ideal
intermediate direction (Adam, 100 steps, converged by 20; edit norm capped at 1.2× the natural change, and the cap
binds at every waypoint;
gradients through blocks 23–24 and the predictor in bf16, re-scored in fp32). They hit it to 0.08°, but only at the
norm cap, with 28% of each edit in the ring plane and forecast radius ≈ 5 (unedited ≈ 0.96). The path radius goes
2.2 → 1.0, never crossing the interior; cosine 0.12 with the spline's edit, 0.00 with the chord's; closer to the spline
in the ring plane (1.35 vs 1.75), slightly closer to the chord in full space (+0.04 [0.02, 0.06]). With an angle-only
objective the predictor can be steered along an off-ring route, so the reverse direction is not recovered here. This
objective is weaker than Goodfire's full pullback objective, which also penalises leaving real behaviour, and the
protocol differs from A.8 in four ways: one path per clip instead of one path shared by 16 carriers at the same source
value, an additive edit instead of a replacement of the top PCs, 64 PCs instead of 32, and a hard norm cap where
Goodfire's released weekdays config uses no norm term (months and age do not run pullback in the released configs at all; only the alphabet config carries A.8's term). The first two plausibly favour an off-ring
route; the cap and the on-ring zero start work against one, so the departures do not all point the same way. The full-space comparison is at equal t,
not A.9's closest-point residual; rescored by closest point, these paths sit 1.19 natural units from both references
(the chord itself is 0.18 from the spline). *With Goodfire's recipe*[^pbg]: 8 pairs at 135–180°, one path per pair
shared by 16 carriers, the top-32 PCs replaced with the rest held, 20 free waypoints from a chord start, L-BFGS with
strong Wolfe, a squared-Hellinger target on the predictor-native direction readout (turned into a distribution by a
fitted softmax), no norm term. The loss falls 13.5 → 2.4 and the forecast ends within about 7° of the ideal
intermediate direction, but the activation path sits 0.61 ± 0.09 natural units from the spline and 0.59 ± 0.09 from
the chord (paired +0.018 [−0.004, 0.040]; intrinsic R² 0.15 vs 0.27, p = 0.10), three times the chord's own 0.21
distance from the spline (p = 0.003, chord closer on 8 of 8), along a route of norm 0.4–1.6 across pairs against the spline's
0.3–0.8. The GPU budget capped each pair at 32 loss evaluations (5 outer L-BFGS steps against A.8's 50), so no pair
converged and the loss was still falling. Those figures were scored against a ring fitted on full-clip activations, while the edited
activation is the carrier's context-only (frames 1–8) point-22 vector, which sits 0.56 from that ring before any edit.
Rescored against a ring fitted on context-only activations of the same knot clips (same PCA-64, centroids, spline and
chord recipe)[^ctx], the carrier mean starts 0.14 ± 0.01 off the ring where the chord starts on it (the chord's path-averaged
distance from the spline is 0.16; individual carriers sit 0.45–0.54 off, and the scored path is the carrier mean; distances in full-clip natural units, on which context-only centroid spacings run 0.61–0.75×),
the full-clip chord initialisation averages 0.57 from it along its length, and the optimised paths average 0.81 ± 0.08
from the spline and 0.81 ± 0.08 from the chord along theirs (end points 0.76 from the spline; paired −0.006 [−0.023,
0.012]; intrinsic R² 0 against both). So the run began about 0.57 off the ring it edits,
because its PCA basis, replaced components, centroids and chord start all came from full-clip activations, and the
optimiser added +0.23 on 8 of 8 pairs; the rescoring changes the ruler, not that anchoring. Rerun with
context-only geometry throughout (PCA basis, replaced components, centroids and chord start all from the context-only
knot activations; same 8 pairs, 16 carriers, 32 evaluations, no pair converged)[^ctx2], the initial path (the carriers' own coordinates outside the top 32, a chord start inside them) averages 0.18
from the spline (0.04 at its first waypoint) and the optimised paths average 0.87 ± 0.07 from the spline and 0.87 ± 0.08
from the chord (start 0.90, end 0.75; optimised minus initial +0.68 [0.51, 0.85], 8 of 8 away; spline − chord +0.000
[−0.019, 0.019]; R² 0 / 0.01; full-clip natural units, and the loss still falling 0.13–0.30 at the last logged step and
0.27–0.57 to the final evaluation), while the forecast lands 6° from the ideal. The optimiser moves off the ring under Goodfire's
recipe as it did under ours; the unedited carriers of the old angle-only test already sit 0.61 from the ring and its
paths average 0.59 further out (20 of 20). Data separation in these pullback runs is not clean: 3–5 of each pair's 16
carriers are knot clips, 5 are test clips, the readout's softmax sharpness was fitted on test clips, and the behaviour
centroids use all clips (`data_separation` in both files); a leak here would favour recovery, and the result is negative. Our earlier angle-only paths score 1.20 / 1.27 on the same ring
(closer to the spline by 0.07, 19 of 20, from a zero-edit start of 0.61). So neither our angle-only test nor an unconverged run of Goodfire's
recipe recovers the ring from the forecast; both find a route off it, and neither is a test the paper would count as
complete. A negative, as run, on both protocols.

**Speed through the predictor (GPU session 3)**[^s3]. Does the direction picture hold for a straight variable? The design:
128 test carriers from the speed set (constant velocity, 0.25–4 m/s) × 4 targets in the contiguous held-out block of
the point-19 speed run (3.05–3.46 m/s). Each carrier was re-rendered at every target as a pixel twin (disk IoU 0.984). The arms are the
smoothing spline, the chord between smoothed knots, the raw-centroid chord, and a null that aims the spline at the far
end of the range (0.25 m/s). Edits were made at point 22 (own norm, the chord's norm, the natural twin change) and point 12 (own, natural). Readouts are fit
on the predictor's own unedited forecasts of the 490 probe clips: a direct speed probe (test MAE 0.18 m/s, R² 0.955)
and a displacement readout from per-step position probes (MAE 0.56 m/s, R² 0.55). The twin's own context reads 0.22
m/s from the target with R = 0.93 [0.83, 1.02] on the direct probe (unedited 1.25 m/s), the analogue of the direction
twin's 0.96. Direct readout, error to target (m/s) / R speed:

| Arm | pt 22 own (0.30–0.42× twin change) | pt 22 at the chord's norm | pt 22 natural norm | pt 12 own (0.37–0.44×) | pt 12 natural norm |
|---|---|---|---|---|---|
| spline | 0.31 / 0.62 | 0.32 / 0.61 | 1.29 / 3.00 | 1.08 / 0.13 | 0.91 / 0.57 |
| chord (smoothed knots) | 0.32 / 0.62 | 0.32 / 0.62 | 1.34 / 3.21 | 1.08 / 0.13 | 0.92 / 0.70 |
| raw-centroid chord | 0.33 / 0.65 | 0.42 / 0.47 | 0.79 / 1.90 | 1.08 / 0.13 | 0.90 / 0.39 |
| null (aimed at 0.25 m/s) | 3.17 / −0.67 | 1.71 / −0.25 | 3.07 / −0.92 | 1.65 / −0.21 | 1.60 / −0.21 |

At point 22 at their own norms the structured arms move the forecast's speed code 62–65% of the way, and the null lands on its own
aim (0.27 m/s from 0.25; null − chord +2.85 m/s [2.81, 2.90]), so the edit is target-specific. Scaled up to the natural twin change
(3.2× the spline's own norm) every arm overshoots (R 1.9–3.2) and lands farther from the target: the twin's change
carries more than speed. The route does not matter. Spline − chord is −0.005 m/s [−0.007, −0.004] at their own norms,
+0.002 [−0.00002, 0.003] at the chord's norm and −0.05 [−0.08, −0.03] at the natural norm, where both overshoot. The
raw chord's differences follow its dose: it is 0.10 m/s [0.08, 0.12] worse than the spline at the chord's norm and 0.50 [0.42, 0.58] better at the
natural norm, where it overshoots less. The forecast's disk positions do not follow. On the displacement readout R is 0.05
[−0.02, 0.11] (spline, own norm) against the twin's 0.48 [0.21, 0.78], though that readout is weak itself. The edited forecast sits 19.6 px from
the twin's true disk, against 20.7 px unedited and 7.2 px for the twin's own forecast. Off target, the forecast direction moves
7.6° [5.4, 10.1] at the spline's own norm, about what re-rendering the clip at another speed does (8.5° [7.4, 9.7] for the twin), and
18–26° at the natural norm. At point 12 the edits' own norms reach the forecast with R 0.13 (1.08 m/s off vs 1.25),
the direction session's wash-out. At the natural norm R rises to 0.57 [0.48, 0.66] for the spline (chord 0.70 [0.57, 0.85],
raw chord 0.39; displacement readout 0.42 [0.28, 0.59]), so for speed the point-12 wash-out is partly a matter of
dose. Direction at point 12 was not rerun at the natural norm. Its one high-dose point-12 edit, the label-free interpolating
spline at 7.2× the twin change, read 77.9°, but that edit was misaimed. Through the encoder (full clip, own norm, a speed probe refit at every later point) point-12
edits keep R 0.90 at point 12, 0.44 two blocks later and 0.13 at point 25. Point-22 edits keep 0.57 at point 25 (0.42 m/s
from target, against 1.27 unedited and 0.10 for the twin). So speed behaves as direction does in where edits survive
at their own norms, and the spline adds nothing over the chord, as the straight geometry predicts (§4.1). The forecast's
speed code moves, but the forecast disk does not move faster, so this is a probe-level change in the forecast and not a
faster predicted disk (`figures/fig_session3_speed.png`).

**Time-reversed clips** (the forward-trained probe read on reversed clips)[^trev]. From point 1 on, the direction probe
reads θ + 180° on the reversed clip: the error to θ + 180° is 20.5° at point 1 and 5.7–10.3° from point 2 on, with
98–100% of clips closer to the flipped angle. The probe tracks motion direction, not position or occupancy (a reversed
constant-velocity clip has the same frame set). A speed probe transfers to reversed clips (R² 0.944–0.981 vs
0.936–0.987 forward at points ≥ 1).

**Straightening across depth** (Hénaff, Goris & Simoncelli 2019; the open question of Musa et al., arXiv 2609.01551;
`figures/fig_straightening.png`)[^str]. Does the encoder straighten a clip's own path through time, as the perceptual
straightening hypothesis predicts for a predictive code? At each point, the whole-frame mean token of each of the 8
tubelets gives an 8-step trajectory, and its curvature is the mean angle between successive displacements (0° for a
straight line, 90° for a random walk, above 90° when successive steps partly reverse). The pixel reference is the
32 × 32 frames averaged into the same 8 tubelets, and straightening = pixel − latent curvature. Clips: 750
constant-velocity and 750 accelerating direction-set clips, 1,536 acceleration-set clips; controls: the random-init
encoder, reversed clips, and random walks with each clip's step norms (isotropic, and matched to the displacement
covariance). It does not straighten. On constant-velocity clips latent curvature is 102.5–118.7° at every point,
against 81.6° [79.8, 83.3] in pixels and 88.7–91.1° for the two random-walk nulls, so the straightening index is negative
everywhere (−20.9° at its least negative, point 6; −36.9° at point 22) and the latent steps zig-zag. The curve dips at
middle depth, to a minimum of 102.5° at point 6 (point 6 in 2,000 of 2,000 bootstrap draws, never inside the paper's
zone 8–12; the zone's minimum, 106.1° at point 9, sits 11.2° [10.9, 11.5] below point 25), then climbs to 117–119°
from point 17 on. The random-init encoder is flat at 115.9–116.3° at every depth, so the mid-depth dip is a training
effect, but it is a partial undoing of a zig-zag the untrained network already has, not a straightening. Accelerating
clips (pixel curvature 54.0°) give 104.2–117.4° with the same minimum at point 6. At points 3–9 faster clips have
straighter latent paths (Spearman −0.41 to −0.52 with speed), while pixel curvature rises with speed (+0.98); from
point 12 on the link is weak (−0.03 to −0.39). Reversing the clip changes the constant-velocity curve by at most 1.6°.
Caveats: the whole-frame mean includes the static background, and 8 tubelets give 6 angles per clip.

**Is the ring's fold used? (saddle axis)**[^sad]. §4.1 found the ring bent out of its plane along a cos 2θ axis. Does
the predictor use that fold? The axis comes from a fit C(θ) ≈ μ + A1 [cos θ, sin θ] + A2 [cos 2θ, sin 2θ] on the train centroids (PCA-64): u is A2's
top direction with the ring plane projected out. It is nearly a pure second harmonic (k = 2 share 0.997 / 0.999) and
holds 0.17 / 0.21 of the centroid variance at points 12 / 22 by this fit, against 0.20 / 0.24 for the third centroid PC
in §4.1. On the 16 headline-arc carriers, the coordinate along u (read on the context-only activation) is scaled by 0
(removed), 2 or −1 at points 12 and 22. The controls are 20 random axes outside the ring plane and u, with the same context-only variance and
the same scalings. The headline smoothing-spline path is also run with and without its component along u. Readouts are
predictor-native probes of the forecast (direction, speed, per-step position) plus a cos 2θ / sin 2θ probe (CV R²
0.76). At point 22 (point 12 in brackets):

- *The forecast carries the fold.* Removing it moves the forecast's cos 2θ readout by 0.37 [0.25, 0.48] in the sign the
  edit predicts (0.063 [0.041, 0.082]). Random axes move it by −0.008 (−0.007), and none of the 20 draws moves it as far.
- *The forecast's direction does not use it.* The error to the true direction (unedited 6.2°) changes by +0.19° [−0.39,
  0.87] on removal, −0.14° [−0.95, 0.53] at ×2 and +0.48° [−0.57, 1.65] at ×−1. Random axes of the same variance move it
  more, +2.6° [1.4, 3.8] and +7.4° [5.3, 9.6], so saddle − random is −2.4° [−3.8, −1.0] and −6.9° [−9.5, −4.4]. The
  forecast position moves less than under a random axis (1.8 px vs 4.3 px; −2.4 px [−3.3, −1.6]), and speed does not move (+0.009 m/s
  [−0.04, 0.06]). The whole forecast still changes by as much as under a random axis (0.11 vs 0.10 of the median
  distance between probe-clip forecasts). At point 12 the direction change is +0.56° [−0.15, 1.28], tied with random
  (+0.42° [−0.30, 1.19]). The axis refit on context-only activations (cosine 0.97–0.98 with the full-clip axis) gives
  the same results.
- *The spline's bend along u is inert.* The component along u is 0.43–0.56 of the point-22 path edit's norm (0.38–0.56),
  yet flattening the path into the ring plane changes almost nothing. The endpoint forecast error changes by −0.10° [−0.96, 0.74] (+0.03°
  [−0.60, 0.68]), the minimum forecast radius by −0.007 [−0.019, 0.005] and the endpoint forecast direction by 1.2°
  [0.6, 1.9]. The edit-point probe does not see it either (+0.03° [−0.04, 0.10]).
- *What the fold tracks.* On train clips, cos 2θ and sin 2θ explain 78% / 84% of the coordinate. Within a direction value it
  correlates with speed at 0.38 / 0.28. Across the 64 value means it correlates with the disk's mean horizontal and
  vertical offset from the frame centre (0.52 / −0.63 at point 12, 0.52 / −0.62 at point 22). A fold driven by frame geometry
  would show this, but any cos 2θ-shaped nuisance would correlate this way across values.

So the fold is real and reaches the forecast. The forecast does not use it for direction, speed or position: removing
or flipping it moves only the forecast's own cos 2θ code, and a random axis of the same variance disturbs direction
more. About half of the smoothing spline's edit on these carriers lies along the fold, and that half is inert for the
forecast. The along-path route effect above used the interpolating spline, which was not flattened, so this test
does not locate that effect. The bf16 forward differs from the fp32 cache by
1.6° [0.9, 2.5] on the direction readout; every contrast above is bf16 against bf16 (`figures/fig_saddle_axis.png`).

**Is the ring's radius read as speed? (radial steering)**[^rad]. On the speed set the ring's radius grows with speed
(§4.1). At point 22 I scaled each carrier's ring-plane radius about the context-only ring centre by ×0.25, 0.5, 1.5 and 2,
keeping the angle fixed. The controls are 20 random 2-planes outside the ring plane and saddle axis, matched in per-axis variance and given the
same scalings, with the same 16 carriers and readouts. On the context-only train clips the mean radius rises from 5.1 at
speed label 0 (the accelerating-from-rest clips) to 7.0 at 7 m/s (clip correlation 0.32). By that curve, ×2
corresponds to +4.4 [3.0, 5.8] speed units, and both shrinks hit its floor (−2.4).

- *Speed does not move.* The forecast speed changes by +0.004 m/s [−0.11, 0.11] at ×0.25, −0.001 at ×0.5, −0.013 at
  ×1.5 and −0.03 [−0.16, 0.11] at ×2 (unedited 2.67). Random planes move it at least as much in 16–20 of 20 draws.
- *The forecast's direction code scales with it.* The forecast's direction-probe radius (unedited 1.03) follows the edit: −0.27 [−0.36,
  −0.17] at ×0.25, −0.19 at ×0.5, +0.21 at ×1.5 and +0.44 [0.33, 0.55] at ×2. No random plane moves it as far at any
  scale (radial − random −0.30 [−0.41, −0.17] and +0.41 [0.25, 0.56]).
- *Direction degrades at both ends, no more than for random planes.* The error to the true direction rises by +8.5° [3.1, 14.0]
  at ×0.25, +3.0° [−1.2, 7.4] at ×0.5, +3.3° [1.5, 5.2] at ×1.5 and +8.2° [4.8, 12.0] at ×2. Radial − random is +5.6°
  [−0.7, 12.5] at ×0.25 and −0.8° [−5.9, 4.3] at ×2.
- *Position moves.* The forecast disk moves 22 px at ×0.25 and 30 px at ×2, against 7 px and 10 px for random planes (+15.0 px
  [10.8, 19.5] and +20.0 px [14.4, 26.0]). Edit norms are matched (radial − random −0.20 [−1.5, 1.0] and −0.27 [−2.0,
  1.3]), and the total forecast change is the same (−0.01 [−0.07, 0.03] and −0.02 [−0.09, 0.04]).

So the predictor reads the ring radius as the strength of its direction code and as position, not as speed. Across
an 8× radius range the speed readout moves no more than under a random plane. The radius–speed cone of §4.1 is a
correlation, and at point 22 on these carriers it is not a speed channel the forecast uses. The limits are one point,
16 carriers, no target (the angle is kept), and a direction-set speed label that puts the 750 accelerating clips at 0,
so the curve that converts radius to speed mixes motion types.

### 4.6 Which tokens carry direction, by depth (token-source patching at the predictor)

The paper's own future-work hypothesis (l.795–798) is that velocity is "bound" to the object in the middle of the
network and that the end of the network is "catered to the optimization objective of predicting the next frame in
latent space". Token-source patching tests this directly. For 64 random pairs and 64 position-matched pairs of
direction clips (same speed, ≥ 90° apart; matched pairs start within 0.15 m), clip B's tokens replace clip A's at
one encoder point, either the disk tokens (the disk masks of A and B plus a one-patch ring, 9–13% of the 2,048
tokens) or the background (the complement), and the rest of the encoder plus the predictor run on the patched
state. The binding fraction is how far the direction readout moves from A toward B (1 = all the way), read by a
held-out ridge probe on the predictor's forecast and, as a check, on the encoder output
(`results/p5_token_patching.json`, `pair_types.{random,posmatch}.readers.{forecast,encoder_output}.points`).

| tokens patched | point 0 | point 8 | point 12 | point 16 | point 22 |
|---|---|---|---|---|---|
| disk tokens, random pairs, forecast reader | 0.99 [0.97, 1.00] | 0.98 [0.96, 0.99] | 0.88 [0.86, 0.90] | 0.76 [0.73, 0.79] | 0.22 [0.20, 0.24] |
| background tokens, same | 0.02 [0.01, 0.04] | 0.04 [0.02, 0.05] | 0.12 [0.09, 0.15] | 0.24 [0.21, 0.27] | 0.77 [0.75, 0.79] |
| disk tokens, position-matched pairs, forecast reader | 0.99 [0.99, 1.00] | 0.98 [0.96, 0.99] | 0.89 [0.87, 0.91] | 0.74 [0.71, 0.77] | 0.18 [0.17, 0.20] |
| background tokens, same | 0.01 [0.00, 0.02] | 0.03 [0.01, 0.05] | 0.12 [0.09, 0.14] | 0.26 [0.23, 0.29] | 0.82 [0.80, 0.84] |

The encoder-output reader gives the same curve within 0.03 at every point. Up to point 12 the predictor reads the
disk's direction from the disk's own tokens: swapping 9–13% of the tokens moves the forecast 88–99% of the way,
swapping the other 87–91% moves it 1–12%. Between points 12 and 22 that reverses. At point 22 the disk tokens carry
0.18–0.22 and the background 0.77–0.84, about twice the disk's token share (0.09–0.13) and near the background's (0.87–0.91), so by the
last blocks the direction code is spread across the whole frame rather than held by the object. The position-matched
pairs rule out the disk's location as the carrier. Read against the paper's C.1.4 hypothesis (velocity "most bound" to the object in the middle layers), the disk-token share falls monotonically from the input, 0.99 / 0.98 / 0.88 / 0.76 / 0.22 at points 0 / 8 / 12 / 16 / 22: there is no mid-network binding peak; the code is object-bound from the start and delocalises late, which agrees with the two-disk result in §5 (binding lowest in the zone) rather than with the hypothesis as stated. No untrained control was run for the patching itself. It also says why edits at point 22
reach the forecast when a per-token object code would not: a pooled edit at 22 lands on the tokens the predictor
actually reads. Provenance: forward on box 1 at commit 8a54162, clean tree; scored locally; 480 probe clips per reader,
5-fold ridge, CV R² 0.91 (forecast) and 0.94 (encoder output). Limits: one object, one render, five points, pair
bootstrap CIs over 64 pairs; a swapped disk also swaps the disk's appearance, which is identical across clips here.

**The same question from the encoder side (pooled object, background and scene tokens)**[^ovs]. Before the
predictor, does the encoder itself keep direction on the disk? Three pools per clip, built exactly from the stored
time-pool, disk-pool and disk-mask arrays: the mean of the disk tokens (object), the mean of every other token
(background) and the stored mean over all tokens (scene). Accuracy cannot tell them apart: at point 8 a ridge probe
reads direction at R² 0.987 from the object pool, 0.978 from the background and 0.979 from the scene (MAE 3.1°, 4.7°
and 4.5°), and at point 22 0.995, 0.991 and 0.992; the untrained copy's scene pool stays at 0.87–0.88 at every point.
What separates them is the axes. A probe fit on the object pool and read on the background pool transfers with R² 0.27
at point 8 and 0.17 at point 12 but 0.77 at point 22 (background→object 0.72, 0.44 and 0.88), and the angle between
the object and background ring planes is 78–89° at points 8 and 12 and 53–56° at point 22. So through the emergence
zone the disk tokens and the rest of the frame both decode direction but on nearly orthogonal directions of the
residual stream, and by the late blocks they converge on one shared code, which is the token-patching result read at
the encoder rather than at the predictor. The object pool's ring at point 8 is also rounder than the scene's (b/a
0.97 against 0.74, saddle share 0.17 against 0.28). Every pool decodes θ + 180° on 99–100% of time-reversed clips from
point 1 on, so this is a motion code and not a trajectory-shape artefact. Limits: the background pool decodes
direction at R² 0.52 already at point 0 (edge pixels below the disk-mask threshold and the 2-frame tubelet leak
motion into "background" patches, so the background is not disk-free input); there is no untrained control for the
object/background split, because the stored random-init extraction has no disk pool; and pooled tokens cannot say
whether the background's code is written there by attention from the disk or reflects a disk-dependent complement
set. The held-out-arc numbers in this file use a clip bootstrap whose intervals often exclude the point estimate
(duplicate clips bias the smoothing spline), so only point estimates are quoted here and none is used above.

## 5. Beyond the three variables

| Question | Result | Source |
|---|---|---|
| Object permanence | Direction decoded from time steps whose frames contain no disk (89 clips; test 15 clips / 22 tokens): MAE 7.5° [5.6, 9.4] at point 8 (visible 5.4°), 6.1° at point 22 (visible 3.8°); shuffled-label null 84.5°, p = 0.001. The random-init encoder, same clips and protocol, does as well: test-clip absent-step MAE 5.9–7.2° across points vs 6.1–13.4° for V-JEPA 2 (null ≈ 90° for both). V-JEPA 2 is ahead only late, by ≤ 1.0° on test clips (points 16–25) and 2.4° / 2.0° pooled at points 22 / 25; it is behind at points 1–12. On visible steps V-JEPA 2 is 5–7° better from point 8 on. So above-null decoding after the disk leaves is attention mixing within the clip (no causal mask), not learned carrying. | `p1a_object_permanence.json` (`random_init.side_by_side`), `fig6_object_permanence.png` (random-init overlaid) |
| Cartesian vs polar | On constant-velocity clips (596), (vx, vy) reaches onset at point 1 and (sin θ, cos θ) at point 2 (difference −1, CI [−1, −1]); block 1 R² 0.929 vs 0.863. Speed set: 0.985 vs 0.855. The one-block "emergence" of direction is the normalisation v/‖v‖. Direct test at block 1: the angle of the (vx, vy) probe's output has MAE 12.1° against 12.3° for the direct (sin, cos) probe, and R² 0.900 against 0.911 once the direct output is scaled to unit length, so the direct probe's lower R² there is its radius, not its angle; the two angles disagree clip by clip by 13.1°. From point 2 the direct probe is better (8.3° vs 11.2°). | `p1a_support_onset_*_meanpool.json`, `fig1d`, `p1a_support_cartesian_angle.json` |
| Direction transfer (held-out context) | Direction probe fit on the direction set, read on the speed set at point 9: MAE 4.4° (source CV 4.0°); 8.7° below 1 m/s, 3.3° at 1–4 m/s. On the acceleration set: 5.8°. At point 1: 10.8° (23.9° below 1 m/s). | `p1a_support_transfer_meanpool.json`, `fig1c` |
| Spatial generalisation | Train on start x < 0, test on x > 0: at block 1 already R² 0.828 / 0.815 across sides against 0.810 / 0.806 within (consistent with the half-frame finding against C.5's "generalize to unseen regions only after the emergence zone", though this is a whole-frame probe split by start side, not a region-of-frame probe); point 9: 0.971 (MAE 4.9°) vs 0.972 within-side. At point 22, mean-pool 0.957 / 0.975 vs disk-pool 0.988 / 0.987 (negative-to-positive / positive-to-negative side). | `p1a_support_spatial_{meanpool,diskpool}.json`, `fig1e` |
| Direction vs speed subspace (paper C.4 method) | Overlap direction←speed 0.0740 at point 8 (random expectation 0.0781, 5–95% band 0.0756–0.0808); 0.0733 at point 9 (0.0723, band 0.0694–0.0740). Direction vs acceleration 0.0762 and 0.0739, inside or at the edge of the band. Direction←speed at point 8 is below the band (more orthogonal than any of the 20 random draws); the other three are inside or at its edge, so the INLP bases are at least as orthogonal as random ones, yet steering direction still moves the speed readout (§3.3 off-target). | `step2_subspace_angles.json` |
| Objective axis | V-JEPA vs random-init at the direction peak: probes needed to reach ≤ 10° MAE 4 vs 10; nested K 88 vs 26. VideoMAE matches V-JEPA 2 on all three variables with the same onsets (§3.1): 4 probes to the bar, nested K 67, peak 0.992. | `objective_axis.json`, `fig5_objective_axis.png` |
| Position sheet | Start (x, y) is decodable; 36-cell centroid PR 7.46 (point 12) / 3.76 (point 19), Procrustes to (x, y) 0.38 / 0.66; spline steering gives no path advantage (§4.3). | `p2_sheet_speed_L{12,19}.json` |
| Velocity sheet (direction × speed, steering) | The sheet is neither cone nor cylinder: ring radius grows with speed to ~1.4–2.4 m/s then flattens (point 12 noise-corrected radius 3.17 at 0.46 m/s, 7.70 at 3.79 m/s; fast ÷ slow 2.43 [2.06, 2.57] against a speed ratio of 8.29; linear-fit intercept 4.38, a cone needs 0); held-out-cell model fit: cylinder 0.804, radius-scaled ring 0.834, sheet 0.843, noise ceiling 0.910. Steering to held-out cells, matched norm, clip-bootstrap CIs: direction at fixed speed, sheet − one ring per speed band −2.47° / −2.58° / −4.02° at points 12 / 19 / 22, sheet − raw chord −0.42° / −1.02° / −1.50°; both variables, sheet − two sequential 1-D edits −1.66° / −2.16° / −4.35° and −0.060 / −0.049 / −0.031 m/s; off-target direction change of a speed edit, sheet − 1-D speed line −1.3° to −1.8°. The untrained copy has no ring (radius ≈ 0.03). A deeper block hold-out, ablations (radius frozen, global speed axis) and a predictor-forecast readout are running (§8). | `p5_velocity_sheet.json` (`shape`, `model_fit`, `steer.*.sheet_minus_*`), `fig_velocity_sheet.png` |
| Within-clip time (step index) | Each of the 8 time steps has its own pooled feature; the clip-mean-subtracted residuals trace a shared curve that explains 0.82 of held-out within-clip variance at point 22 (a straight line in t 0.15; `geometry["speed/vjepa2/timepool"]["22"].heldout_var_explained` centroids 0.823, line_in_t 0.151). The coordinate advances 1.00 per step in slow, mid and fast bands alike (fast ÷ slow 1.005 [1.002, 1.009] at point 22, where a distance counter predicts 6.02), so it is a speed-invariant frame count, not an odometer. Steering to a held-out step: time-probe error 0.106 (spline) / 0.082 (raw chord) steps at point 22 against 2.43 unedited, spline − chord +0.024 [0.023, 0.025]; side effects ≤ 0.007 m/s and ≤ 0.29°. Control: the untrained copy also decodes t (test R² 0.9735 at point 22, `controls.speed.random.timepool.decode.22.test.t_r2`), so the step index itself is available to any encoder, plausibly as a RoPE- or content-derived time index (V-JEPA 2 adds no position vector to the residual stream); what is specific to training is the shared low-dimensional curve (untrained: 0.007 of within-clip variance) and the exact 1.00 slope (untrained 0.73). At a fixed frame rate the design cannot separate frame count from elapsed seconds, and the time-rescaled "clock" stimuli are speed changes in disguise (`CLOCK_NOTES.md`, #251/#277). Static-disk control not run. Predictor-level test (does a time edit advance the forecast?) running (§8). | `p5_time_manifold.json` (`geometry`, `clock`, `decode`, `steer`, `controls`), `p5_time_manifold_controls.json`, `p5_clock_test_linearity.json`, `results/CLOCK_NOTES.md` |
| Relational motion (two disks) | 637 two-disk clips balanced over common velocity c and relative velocity v_rel. A linear v_rel probe is by construction the difference of the two single-disk probes and gives no evidence of a relational code: V-JEPA 2 0.95 (point 2) vs untrained 0.98, and the V-JEPA readout fails the Galilean transfer across c halves (R² ≤ 0 vs 0.80 untrained). What training adds is identity-blind: relative speed |v1 − v2| reaches OOF R² 0.969 [0.964, 0.973] at point 17 (untrained best 0.204, pixels 0.00; `headline.best_by_target.abs_v_rel`), v_top − v_bottom 0.84, while v1, v2 and signed v_rel fall to 0.58–0.59 / 0.16 at the output; an MLP on activations beats the same MLP on the decoded single-disk velocities for |v1 − v2| from point 4 (+0.63 to +1.12 held-out R²). Per-disk pools contradict the paper's C.1.4 binding hypothesis in this setting: velocity is least bound to its own disk in the emergence zone (index 0.13 at points 8–9 vs 0.4–0.7 at point 1 and 0.21 at 20–24); measured against the other disk's tokens the untrained copy's index is ≈ 1, so training spreads one disk's velocity into the other disk's tokens; measured against the background instead (the object-vs-scene convention in §4.6) the untrained index is ≈ 0 at every depth while V-JEPA 2's rises with depth (0.12 at point 8, 0.39 at 22), so the verdict on "bound" depends on the comparison set, and the zone value is a plateau over points 6–17 rather than a minimum Caveats: CIs resample clips not cells; the diskmask covers one disk; late-layer v1/v2 readouts are weak, which is what makes the composition null fail. | `p5_relational_motion.json` (`headline`, `sources`, `beyond_composition`, `binding`), `p5_relational_stimuli_validation.json`, `fig_relational_motion.png`, `fig_relational_binding.png` |

## 6. What this says about her framing

- **Detection vs use.** Detection is easy on this data. A random network, and random features of a 32-number
  trajectory, detect all three variables at R² ≥ 0.85. Use is where the evidence thins. The Part 1 edit moves a
  held-out linear probe, but whether it beats a rank-2K random basis depends on the evaluation probe (from N = 4
  under the near-unregularised probes, N = 9 under mine, 200 draws; a rank-matched one from N = 2–7), it works in an untrained network too, and it
  leaves an MLP on disjoint clips 17–23° off. On the ladder in `PART2_RATIONALE.md` §2, this project reaches rung 3 at
  the steered layer and rung 4 only by circular measures. Rung 5 was tested: at point 22 and at the encoder output the
  edit reaches the predictor's forecast; at points ≤ 12 it washes out and does not (§4.5). At point 22 the forecast
  passes through the intermediate directions on the spline and jumps on the chord (−13.4°); at the encoder output the
  chord lands as close as the spline or closer.
- **Internal world model vs stimulus-response, and which readout to trust.** For this stimulus class the random-init
  control shows that pooled linear availability is architectural and that carrying direction into disk-free tokens is
  not a training effect. The per-patch probes show what training does add: the random network never gets past a mean
  per-position R² of 0.39, the regime of fragmented local signal that pooling adds up, while V-JEPA 2 reaches 0.96 by
  block 6 on the supplied clips and, on a harder stimulus, forms that code with its largest rise at points 4 → 6 on every render seed and, across the paper's depth, loses
  half-frame transfer and jumps back to chance, which the random network never does (§3.1). So the emergence zone is a claim about the per-patch readout, which the paper's C.5 says, and a mean-pooled
  curve can neither confirm nor refute it. VideoMAE matches V-JEPA 2 on every pooled Part 1 measure (I did not run it
  per patch), so none of this is specific to latent prediction or shows that the variables are used to predict. The authors' OpenReview response states that "all 13 models encode motion direction
  (R²≥.43), regardless of objective", classification CNNs included, so availability is their own finding; training buys
  precision, fewer probes to steer (4 vs 10 to reach 10°[^obj]) and a label-free ring.
- **The linear representation hypothesis: right about the subspace, wrong about the moves.** Direction lives in a 2-D
  linear subspace (sin, cos) with a ring on it, and the paper's §7.1 says "manipulating only the unit-circle subspace does not effectively steer direction". An edit built from one
  probe does steer once it is covariance-weighted (3.2°, §3.3), but that edit leaves the probe's plane, the same
  construction on a random 2-D subspace reaches a median 4.2°, and her unit circle sits in the MLP units, not the
  residual stream tested here; so the 2-D subspace is where the code lives, and what the learned probe buys is
  specificity, not reach. The
  steering corollary is what fails geometrically: in the ring plane the straight path between distant directions
  crosses the empty interior (readout radius 0.61) where the curved path does not (0.86), though in the 64-D edit
  subspace it is no farther from real clips (§4.3). Independent readouts at the steered layer do not care; the
  predictor's forecast at point 22 does, in angle only, and at the encoder output it does not (§4.5). The supported statement is "for a cyclic variable the
  hypothesis describes the subspace and misdescribes the moves, geometrically".

## 7. Limitations and next steps

- **Stimulus.** A single disk on a flat background is nearly pixel-decodable. The half-frame dip and jump at the paper's
  depth appear only on the hard rendered set (§3.1), which has 392 clips per render seed and 8 directions, covers
  direction only and reuses 7 start positions across all (θ, v) cells; the sharp per-position rise does not appear on
  any seed. So it is a partial reproduction on one small stimulus family, not on the supplied data.
- **Per-patch probes.** Features are averaged over the 8 time steps at each position, so time structure within a
  position is not probed; the half-frame test is one pooled probe per half; the rendered sets were read at 10 points
  only (13 on hard seeds 1–2), so their onsets of 4 and 6 are upper bounds. VideoMAE and speed were not run per patch.
- **Training dynamics.** With intermediate V-JEPA 2 checkpoints, the random-init vs final contrast becomes a curve.
  That is the natural test of when precision and the ring appear.
- **Predictor readout is a probe; the edit overshoots.** A probe of the pooled forecast, not a rendered future;
  point-22 edits overshoot in position (R 1.3–1.8); one stimulus, four targets in one 45° arc. The spline's lead is in
  angle only and the smoothing spline (the Part 2 default) is the worst real arm at the endpoint (over the path the chord is).
- **The late causal follow-ups are small.** The saddle and radial tests use 16 carriers on one held-out arc, one
  draw of 20 random axes or planes, and points 12 / 22 (radial: 22 only). Session 3 steers speed on one held-out block
  (128 carriers × 4 targets), and its speed edits change the forecast's speed code but not the forecast disk's motion.
  The twin-difference arm is one fit set (seed 0, 2 clips per value) at point 22. All four files were scored from a
  dirty worktree at 3c13095 (§4.5).
- **The hollow is in the ring plane and the whitened direction subspaces.** In the unwhitened 64-D edit subspace and
  full space the chord is no farther from real clips than the spline (§4.3); once the variance a direction reader
  ignores is whitened away, its 180° midpoint lies 8.9× (point 12) and 10.8× (point 22) the real clips' 5-NN distance
  out against 0.9× / 1.4× for the spline, but that density run uses all-value splines, not held-out ones (§4.3). Along the path the forecast follows the intermediate directions along the spline and jumps
  along the chord (−13.4° paired, −31.3° at large shifts); the reverse test does not recover the ring under our protocol or an unconverged run of Goodfire's (§4.5).
- **Post-hoc verdict rule.** Iterated after seeing results, frozen at 8d3cac8 before the arc sweep (§4.2); the gaps and
  CIs are the evidence.
- **Label-free coordinate.** Found only at point 12 and only through my centroid-plane fallback; Goodfire's own
  label-free angle fails its periodicity test at point 12, and point-22 steering and session 2 at points 2, 8 and 22 use
  the labels, as only Goodfire's 70B cyclic runs do (§4.1).
- **Encoder-output steering.** At point 25, the predictor's input and Goodfire's site, edits reach the forecast
  (chord 12.5°, interpolating spline 15.7°) but the spline's matched-norm lead over the chord from point 22 does not
  carry over (+10.8° worse at the chord's norm, 2.1° better at the natural norm; §4.5). The route effect in the forecast
  is a point-22 result.
- **Conceptor comparison is one arc, outside COAST's regime.** COAST's aperture rule found no aperture in its overlap
  band at either point, its pseudoinverse AND is invalid for 87–90% of the steered pairs (Jaeger's AND used instead),
  the target-aimed arm is my variant, and the 16-arc aggregate was not run (§4.4). The negative is for conceptor
  steering of a ring code, not for COAST on its success/failure task.
- **Off-target readouts are transferred probes.** The speed probe is fit on the speed set and is 0.56 / 0.44 m/s off on
  unedited direction clips; off-target changes are read as ratios to natural spread, and spline and chord are compared
  on unpaired intervals (§4.4).
- **New Part 2 files lack clean provenance.** The conceptor, off-target, FFT, density, held-out two-route and
  straightening scripts were uncommitted when these results were written; the density and conceptor files record a
  dirty worktree at 3c13095, the two-route files an unknown commit, and the FFT, off-target and straightening files no
  commit.
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
  `session2_extras.py`, `session2_native_readout.py` (predictor-native probes), `session2_norm_matched.py`, `session2_along_path.py` (forecast along the path, reverse test; `session2_predictor_along_path.json`, `session2_reverse_path.json`). The 70/30 reruns set `WM_SPLIT_PATH=splits/split_paper70.json`. Second look at Part 2 (commit 33a2cbd, clean): `run_ring_occupancy.py`, `run_isometry_linear.py`, `summarize_shift_dependence.py`; `PART2_SECOND_LOOK.md` is the audit record against the Goodfire paper. Part 1 follow-ups: `p1a_perpatch.py` (GPU extract + per-patch probes; `p1a_perpatch_direction_*.json`), `run_paperscale_velocity.py`, `run_support_cartesian_angle.py`, `run_audit_robustness.py --items 1 2 3 4` (grouped CV, raw coordinates, sawtooth metrics, evaluation-probe recipe), `run_stop_rules.py --items 1 2` (stop-rule sweep and one-column removal). Part 2 follow-ups: `session2_encoder_output.py` (point 25 and the point-12 labels-order spline), `run_angle_goodfire.py` (Goodfire's periodicity test, `--no-holdout` for all 64 centroids), `run_isometry_linear.py --angle labels`. Comparison and geometry follow-ups (§4.1, §4.3–§4.5): `run_conceptor.py --layer {12,22}` (point 22 with `--labels-angle`), `run_offtarget.py --layer {12,22}`, `run_fft_harmonics.py`, `run_local_density.py --layers 8 12 22`, `run_two_route_heldout.py --layers 12 22 --seeds 1 … 16`, `run_straightening.py`.
- **Provenance.** Every results JSON records the split sha256 (`98e6310c…`), seeds (split 0, all others 0), git
  commit and a dirty flag. The 20 `p2_steer_*` files and all 32 arc runs were produced at commit 8d3cac8; the other
  Part 2 files at 677b305, 8e552c1, 8829195, 8f08444 or fbf4f72; all with `git_dirty_src_or_scripts: false`. Most
  Part 2 files record split and source paths inside the frozen scratchpad worktree that ran them; the split sha256 and
  the commit are the same as the repository's. The session 2 files were scored at b9c53d0 or 0f34ec2 (the native
  readout at 8734f4b, the norm-matched rerun at 494afe2, along-path and reverse at 21b27b6); the Goodfire-recipe pullback at d0e459c (dirty scripts, committed as cab0dfa) and its context-ring rescoring at 03c6b2b (clean) with the dirty flag set;
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
  time with no separate cost recorded (`session2_encoder_output.json`, `forward.seconds_total`); the Goodfire-recipe
  pullback ≈ 62 GPU-min and the context-only extraction 2.0 GPU-min on the same box at $0.198/h (≈ $0.21, not in the
  $1.19 total). No other box's cost is recorded.
- **Tests.** `pytest --collect-only` collects 221 tests at the commit of this report.

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
[^bake]: `results/p2_bakeoff_direction_direction_L{12,22}_contiguous.json`; the raw-centroid row from the rerun `results/p2_bakeoff_direction_direction_L{12,22}_contiguous_rawchord.json` (`chord_raw.err_probe_matched`, `err_mlp_matched`; unmatched 4.7° / 3.6°, the headline endpoint). Rescaled to the spline's norm the raw chord loses 1.7° at point 12 and 4.3° at point 22 relative to its unmatched endpoint, and at point 22 the smoothed-knot chord then reads 0.5° better than it.
[^s2]: `results/session2_plan.json` (`targets`: 4 per carrier, all in 303.75°–343.125°), `results/session2_renderer_validation.json`, `results/session2_stimuli_validation.json`.
[^fig22]: `loose_threshold` in each `p1b_*` file: R² < 0.3 for direction and R² < 0.1 for speed and acceleration, the thresholds of the paper's Fig. 22. For speed and acceleration the K at that threshold equals the nested K at every V-JEPA point except acceleration at onset (493 vs 466). `K_loose_censored` is set only in the VideoMAE files (true for acceleration at point 22, where K_loose = 87 is a floor); the V-JEPA files predate the flag, and there speed and acceleration K_loose is also a floor at every point (the MAE rule stops the sequence while R² is still 0.13–0.20) except nested acceleration at onset, where R² reaches 0.071 (`rounds[].cv_r2`, `paper.rounds[].test_r2`).
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
[^interp]: `results/p2_interp_labels_summary.json` (`points.{12,22}.heldout_reconstruction.{labels_order,stored_unsupervised_order}.contiguous`, `seed0_interp_labels`, `arcs_interp_labels`, `delta_ratio_spline_over_chord`), `results/p2_interp_labels/`, `results/arcs_interp/`.
[^ext]: `results/p2_extrapolation_linear_ext.json` (`runs.*.{smoothing_linear_ext,interp_linear_ext_goodfire_code,stored_smoothing_cubic_ext}`), `results/p2_linear_ext/`, `results/p2_linear_ext_interp/`.
[^iso]: `results/p2_isometry_goodfire_method.json` (`layers.{8,12,22}.new.{interp,smooth}.{geo,lin}_pearson`).
[^isog]: `results/p2_isometry_goodfire_coord.json` (`layers.{8,12,22}.goodfire_angle.{interp,smooth}.{geo,lin}_pearson`, `goodfire_periodicity_test`, `angle_vs_labels`, `geo_below_chord_goodfire_angle`; 213 tests at ee828a7).
[^isof]: `results/p2_isometry_goodfire_full.json` (`layers.{8,12,22}.{goodfire_full_angle,goodfire_angle,labels_angle,unsupervised_angle}.{interp,smooth}.{geo,lin}_pearson`, `layers.*.bootstrap.variants.*` with percentile / shifted / basic intervals, bias and calls, `geo_minus_lin_calls`, `branches`, `circular_corr_with_labels`); 8ca50cf, ce42042.
[^pbg]: `results/session2_pullback_goodfire.json` (`goodfire.per_pair[].{resid_to_spline,resid_to_chord,r2_spline,r2_chord,chord_baseline_resid_to_spline,loss_chord_init_fp32,loss_final_fp32,n_evals,outer_steps,carrier_ctx_vs_fullclip_pca32_offset_over_unit,path_norm32_over_unit_by_t}`, `goodfire.summary_{64d,32d}`, `old_reverse_test`; provenance records a dirty worktree at d0e459c for the scripts, committed as cab0dfa); `scripts/session2_pullback_goodfire.py`; box 53030966, ≈ 62 GPU-min; cab0dfa, 216 tests. The point-25 along-path read was not run (GPU budget).
[^ctx]: `results/session2_pullback_goodfire_ctxring.json` (`goodfire.per_pair[].{resid_to_spline,resid_to_chord,r2_spline,r2_chord,chord_baseline_resid_to_spline,carrier_offset_from_ctx_ring_over_unit,carrier_offset_from_fullclip_ring_over_unit}`, `init_fullclip_chord_path`, `goodfire.summary_{64d,32d}`, `old_reverse_test`, `ctx_ring_provenance`, `data_separation`); context-only activations of 805 clips extracted on the box (2.0 GPU-min forward), `artifacts/session2/ctx_ring/` with sha256s; `scripts/session2_pullback_ctxring.py`; clean worktree at 03c6b2b, c7d1a66, 219 tests.
[^ctx2]: `results/session2_pullback_goodfire_ctx.json` (per pair start / path-mean / end residuals of the optimised and initial paths to the context-only spline and chord, `r2_*`, `n_evals`, `outer_steps`, losses; `summary_64d`, `init_ctx_chord_path`, `old_reverse_test.zero_edit_baseline`); 58.5 GPU-min on box 53030966; clean worktree at d58e6f2, dd47c4b, 221 tests.
[^isol]: `results/p2_isometry_goodfire_labels.json` (`layers.{8,12,22}.labels_angle.{interp,smooth}.{geo,lin}_pearson`; `unsupervised_angle` rows reproduce the label-free figures; `geo_below_chord_labels_angle` false at every point).
[^pp]: `results/p1a_perpatch_direction_{vjepa2,vjepa2_constvel,random,vjepa2_hard,vjepa2_paper_layout}.json` (`curves.{perpos_mean_r2,pooled_mean_r2,pooled_frac_ge_0.5,cross_half_r2,meanpool_r2}`, `onsets.*`, `layers[].halves` for the cross-half MAE, `methods`, `provenance.time_averaging`); `figures/fig1g_perpatch_direction.png`, `fig1h_perpatch_heatmaps.png`; rendered-set layout (7 shared starts) in `results/session2_stimuli_validation.json` (`layout.start_rule`) and `scripts/render_hard_stimuli.py`.
[^hfolds]: `results/p1a_perpatch_hard_folds.json` (`sets.hard` for the stratified folds and `folds_start_grouped.hard` for start-grouped, each with per-point fold means ± SD and `summary.*.onset_per_fold`; `sets.paper_layout` alongside), cc41a6c.
[^adamb]: `results/p1c_direction_L9_adam_basis.json` (per-N error to target / to true, both nulls with p, `n_to_10deg`, `K`, `comparison_vs_ridge_basis` incl. `null_dimension_note`); basis weights `artifacts/inlp/direction_direction_L9_adam_b64.npz` (sha256 in the JSON); `scripts/run_step3_adam_basis.py`; 36c4cdc, 94c96c7, baa8eac, 7dd1eb5; 216 tests.
[^r2cov]: `results/p1c_direction_L9_rank2_covweighted.json` (Euclidean N = 1, covariance-weighted and oracle edits with error to target / to true, off-target speed and edit norm; the 20-draw Σ-weighted null with p; `evalprobe_variants` for α = 1e-3, Adam and split-half; Ledoit-Wolf check); `scripts/run_step3_rank2_covweighted.py`, `src/wm/steer.py:cov_weighted_delta`.
[^n200]: `results/p1c_direction_L9_nulls200.json` (`bases.{ridge,adam}.all_n[]` with learned error and `nulls.{rank_matched,rank_2K}` mean ± SD and exact p over 200 draws, seed 0, draws 1–20 reproducing the stored files; `bases.*.first_n.{rank_matched,rank_2K}.p_lt_0.05/0.01.{first_n,first_n_sustained}`; `evalprobe_variants` at every N); `scripts/run_step3_nulls200.py`; 555b411, 221 tests.
[^sig]: `results/p1a_sigmoid_onset.json` (`curves.*.{inflection_point,inflection_frac,fit_r2,fit.lo,rise_fit_pp,rise_obs_pp,peak_above_chance_pp,accept_rebuttal_rule,within_1_point_of_paper_depth}`, `fold_bootstrap`, `with_point0` for the fits that included the patch embedding, provenance quoting the criterion from `sonia_joseph.md` l.153–154); `src/wm/sigmoid_onset.py`; 555b411, 7921bbb.
[^seeds]: `results/p1a_perpatch_hard_seeds.json` (`by_point[].{perpos_mean_r2,cross_half_r2}.per_seed`, `onsets.{0,1,2}`, `jump_8_to_9`, `extra_point_curves`), `results/p1a_perpatch_direction_vjepa2_hard_seed{1,2}.json` (`layers[].halves.cross_r2_mean`), `results/p1a_perpatch_direction_random_hard{,_seed1}.json`, `figures/fig1j_perpatch_hard_seeds.png`; render seeds at 2abb3e9, extraction on the box from a frozen worktree (GPU forward 611 s), 3a8d7c7. The random-init control is one weight draw (seed 0) on two render seeds.
[^appb]: `results/p1a_appB_sweep.json` (`variables.{direction,speed}.{per_point[].{sweep_cv_mean,sweep_cv_sd,lr,wd,grid},sweep_onset,sweep_onset_ci,ridge_onset_stored}`); 100 / 50 epochs from C.11 and batch 64 (our choice; neither appendix gives one); coupled L2 weight decay, App. B not saying Adam or AdamW; the best of the 20 configurations is selected on the same folds it is reported on, as our ridge α is over 13 values; fbdc60a, 212 tests.
[^gcv]: `results/p1a_grouped_cv.json` (`sets.{direction,speed}_{vjepa2,random}.{stratified,direction_grouped,start_grouped,speed_grouped}.{onset,onset_ci}`; `sector_grouped` with its pooled-prediction `score`); `figures/fig1i_grouped_cv.png`.
[^psv]: `results/p1a_paperscale_velocity_only.json` (`summary`, `models.vjepa2.n392_velocity.onset_per_seed`).
[^raw]: `results/p1b_raw_coordinates.json` (`cells.{direction,speed}_L{8,9}.{raw,stored_zscored}.{nested_K,paper_K}`).
[^saw]: `results/p1b_sawtooth_metrics.json` (`cells.{direction,speed}_L{8,9}.{ridge_nested,ridge_paper,adam_b64,adam_full}.stats.{r2,bacc8,acc15}`); consecutive readout angles from `sawtooth.mean_consecutive_angle_deg` in `results/p1b_direction_direction_meanpool_L{2,8,9,22}.json`; the paper's teeth read from its Fig. 23 by eye.
[^evp]: `results/p1c_direction_evalprobe_recipe.json` (`stored`, `recipe_results.{ridge_alpha_1e-3,adam_c11}.{eval_probe,n_to_10deg,random_basis_p,rank_matched_p}`); the stored floor N from `random_nulls.rows[].random_basis.empirical_p_to_target` in `results/p1c_direction_L{9,22}.json`.
[^enc]: `results/session2_encoder_output.json` (`per_arm.*.{unmatched,chord_norm,natural_twin_norm}`, `spline_minus_chord`, `spline_minus_chord.point22_stored`, `unedited`); `scripts/session2_encoder_output.py`.
[^il12]: `results/session2_interp_labels_L12.json` (`spline_labels`, `stored_point12.spline`, `stored_spline_edit_norm_recomputed`).
[^twd]: `results/session2_twin_difference_L22.json` (`fit_set.{n_fit_clips,n_fit_pairs,k_nn,svd_energy_top_r}`, `per_condition.{chord_norm,natural_twin_norm}.{twin_r1,twin_r2,twin_r4,twin_r8,twin_full,spline,chord,probe_qr}.{dir_err_to_target,R_dir_real_change,px_err_to_twin_true,enc_out_err_to_target,nearest_real_R_pt25,R_act_twin_pt25}`, `twin_minus_best_comparison.*` (best comparison = `spline`, the interpolating spline of the norm-matched rerun; best twin = `twin_full`), `unedited.dir_err_to_target`, `parity`); predictor readouts of the comparison arms from the norm-matched cache (`definitions."comparison arms"`); `scripts/session2_twin_difference.py`; 987 GPU-s (`forward.total_seconds`); scored at 3c13095 with `git_dirty_src_or_scripts: true`.
[^s3]: `results/session3_speed_predictor.json` (`design.{holdout,renderer_validation_speed}`, `n_carriers`, `readouts.{direct,displacement}.{test_mae_mps,test_r2}`, `twin_reference.{direct,displacement}.*`, `twin_reference.{twin_dir_change_deg,twin_px_to_twin_true,unedited_px_to_twin_true}`, `predictor.{12,22}.{own,chord_norm,natural_norm}.{spline,chord,linear_raw,null}.{direct_err_to_target,direct_R_speed,direct_err_to_far_end,displacement_R_speed,px_to_twin_true,dir_change_deg,applied_norm_over_natural_median,scale_median}`, `paired.{12,22}.*`, `propagation.{12,22}.{spline.R_speed,spline.err_to_target,unedited_err_to_target,twin_err_to_target}`; definitions in `keys`); `scripts/session3_speed_predictor.py`; 1,796 GPU-s on box 53030966 (`compute.gpu_seconds_total`), cost not recorded (`compute.cost_usd` null); scored at 3c13095 with `git_dirty_src_or_scripts: true`.
[^sad]: `results/p5_saddle_axis_L{12,22}.json` (`config.plan.points.{12,22}.{full_clip_axis.{share_u,u_harmonic_share_k2},path_bend_fraction_abs_delta_dot_u_over_norm_by_t}`, `config.forward.geometry.*.cos_u_ctx_vs_full`, `readers.cos2theta_sin2theta_stepmean_train_clips.cv_r2`, `result.unedited`, `result.{saddle,saddlectx}.{x0,x2,xm1}.{d_dir_err,d_c2_aligned,d_speed,pos_shift_px,forecast_change_rel}`, `result.random_axes.*.*.{per_carrier_mean_over_draws,frac_draws_abs_ge_saddle,paired_saddle_minus_random}`, `result.path_bent_vs_flat.{flat_minus_bent,edit_point_probe}`, `result.byproduct_diagnostics`, `result.parity`); `scripts/run_saddle_axis.py` (`score`); 16 carriers = the headline-arc carriers of the repair-attribution run, target 320.625°; one forward of 712 GPU-s on box 53030966 shared with [^rad]; scored at 3c13095 with `git_dirty_src_or_scripts: true`. The third-PC shares 0.20 / 0.24 are `p2_ellipse_direction.json`'s.
[^rad]: `results/p5_radial_steering_L22.json` (`ring_radius_by_speed_ctx_train`, `unedited`, `radial.{r025,r05,r15,r2}.{d_speed,d_forecast_radius,d_dir_err,pos_shift_px,implied_speed_change_from_radius}`, `random_planes.*.{d_speed,d_forecast_radius}.frac_draws_abs_ge_radial`, `radial_minus_random.*`); `scripts/run_saddle_axis.py` (`radial`); direction-set speed labels: 0 for the 750 accelerating clips, 1–7 m/s for the 750 constant-velocity clips (`speed_mps` in the direction set's metadata, `vjepa-physics-takehome-4E00/data/direction`).
[^stop]: `results/p1b_stop_rules.json` (`cells.*.{nested,paper}.{K,stored_stop_trigger,r2_at_stored_stop}`, `common_r2_stop_points_8_9`).
[^onecol]: `results/p1b_one_column_removal.json` (`cells.direction.{two_column_stored,one_col_alternate,one_col_top_sv}.{ridge_nested,ridge_paper,adam_b64,adam_full}`).
[^rawchord]: `results/p2_steer_direction_direction_L{12,22}_contiguous_rawchord.json` (`summary.{manifold,linear,linear_raw}.overall.probe_err_to_target`, `gaps.manifold_minus_linear_raw.probe_err_to_target.{mean_over_pairs,se_over_pairs}`, `sagitta_per_target[i].{sagitta_smoothed_chord,sagitta_raw_chord}`); `scripts/run_part2.py:subspace_arms` (`linear_raw` = piecewise-linear path through the raw kept centroids in knot order, same neighbours and weights as the line arm; `src/wm/manifold.py:raw_knot_curve`). Same carriers, split, K = 50 and edit rule as the stored run.
[^adamsc]: `results/p1b_sameclip_adam_direction_vs_speed.json` (`cells.L{8,9,19,22}.comparison.{C11,fig22}`, `cells.*.{direction,speed}.b64.{K_first,c11_first_trigger,K_fig22_censored}`, `cells.*.*.full`, `cells.*.*.stored_cross_set`); `scripts/run_sameclip_adam.py`. Same 750 clips, 596 train / 154 test, per-feature z-score from the restricted train rows, C.11 Adam probe each round (lr 1e-3, weight decay 1e-4, 100 / 50 epochs), batch 64 (full batch at points 8 and 9), paper protocol, K = probes before the first at-chance round.
[^sameclip]: `results/p1b_sameclip_direction_vs_speed.json` (`cells.L{8,9,19,22}.comparison.{nested,paper}_{C11,fig22}`, `cells.*.{direction,speed}.stored_cross_set`); `scripts/run_sameclip.py`.
[^fft]: `results/p2_fft_harmonics.json` (`direction_vjepa2.{point}.{k1,k2,k3,k4plus,ci95,shuffle_noise_power_over_real,shuffle_fractions}`, `direction_random.*`, `speed_vjepa2.12`, `ramp_reference`); `scripts/run_fft_harmonics.py` (200 stratified clip bootstraps with the PCA fixed; 20 label shuffles). The file records no commit; script uncommitted at the time of writing.
[^trh]: `results/p2_two_route_heldout_L{12,22}.json` (`over_arcs.{antipode_in_arc,midpoint_in_arc}.{via_plus90,via_minus90,chord,chord_raw}.{probe_intermediate_mass,probe_ordering,mlp_intermediate_mass,probe_radius_min,probe_radius_start,end_probe_err,probe_mid_err_to_route_mid,probe_intermediate_mass_{plus,minus}_half}`, `over_arcs.*.contrasts.via_plus90_minus_chord_raw.end_probe_err`, `per_arc_info.*.angle_source`); `scripts/run_two_route_heldout.py`. Run in a separate worktree (`/workspace/wm_route`); split sha256 matches, `git_commit` recorded as unknown.
[^dens]: `results/p2_local_density.json` (`layers.{8,12,22}.density.{train,probe}.{full,pca64,leace2w,lda8w,chart2}.{90,135,180}.{chord_over_real_geomean,spline_over_real_geomean,chord_minus_spline_logratio,chord_minus_spline_ci95,frac_chord_beyond_real_p95,frac_spline_beyond_real_p95}`, `layers.*.intrinsic_dimension.*.all.k10.id`, `layers.*.angle_source`, `layers.*.verdict`); `scripts/run_local_density.py`; commit 3c13095 with `git_dirty_src_or_scripts: true` (script uncommitted).
[^coast]: `results/p2_conceptor_direction_L{12,22}.json` (`arms.{manifold,linear_raw,coast_a,coast_a_b0.3,coast_a_centered,coast_b,coast_b_dose_matched}.{endpoint_err_deg,delta_norm_ratio_to_raw_chord.ratio_of_means,off_target.{abs_change_speed_mps,abs_change_start_position_m,ring_plane_energy_frac},trace_C_mean,unsteered_err_deg}`, `aperture.{mean_overlap,selected,in_band,pinv_AND_invalid_frac_by_alpha,pinv_AND_invalid_frac_steered_pairs,and_mode_used}`, `random_projector_null.{random_a,random_b}.probe_err_to_target.{draws_mean,conceptor_same_clips,frac_draws_conceptor_beats}`, `reproduction_check`, `off_target_readouts.start_pos_mean_dist_to_centroid_m`); `scripts/run_conceptor.py`, `src/wm/conceptor.py` (formulas and page numbers of `refs/coast.txt` in the module docstring; Jaeger's AND quoted from memory there). Point 12 on the label-free angle, point 22 on the labels; `--lite` (no β = 0.1 arm, no α sweep for the aimed arm); commit 3c13095 with `git_dirty_src_or_scripts: true` (scripts uncommitted).
[^offt]: `results/p2_offtarget_direction_L{12,22}.json` (`arms.{spline,chord_smoothed,chord_raw,probe_qr,probe_qr_norm_matched,random_curve}.{speed,start}.{mean,ci95,ratio_to_natural_spread,ratio_ci95,signed_mean_mps}`, `natural_spread`, `readout_quality.speed_probe_on_direction_test_velocity_clips_{mae_mps,r}`, `regeneration_checks.max_rel_diff`; at point 22 `identical_within_1e-6` is false, max relative difference 1.7e-6); `scripts/run_offtarget.py` (start position = metres × 32 px/m). The file records no provenance block of its own; script uncommitted.
[^str]: `results/p5_straightening.json` (`latent_curvature_by_point.{constvel,accel_direction_set,random_init_constvel,null_isotropic_constvel,null_covmatched_constvel}.mean`, `pixel_curvature`, `straightening_index_by_point.constvel`, `zone_test.constvel.{argmin_point,boot_argmin_counts,zone_min_minus_point25_deg}`, `reversed_minus_forward.constvel`, `geometry_links_by_point[].constvel_spearman_curv_speed`, `pixel_links.constvel_spearman_pixelcurv_speed`); `scripts/run_straightening.py` (2,000 clip bootstraps). The file records no commit; script uncommitted.
[^ovs]: `results/p5_object_vs_scene_direction.json` (`direction.probe.{object,background,scene,random_scene}[point].{r2,mae,ci}`, `direction.transfer[point]` (row = fit pool, column = test pool; the point-8 matrix is also spelled out in `summary`), `direction.chart_plane_angles[point].object_vs_background`, `direction.geometry`, `direction.timerev[point].*.frac_decoded_closer_to_theta_plus_180`, `direction.heldout`, `binding.curves.*.zones`, `not_computable`, `provenance` (commit cbd0b38, `git_dirty_src_or_scripts: true`; activation hashes match box 1's `sha256_box.txt` for all 10 hashed files); `scripts/run_object_vs_scene.py` (`--binding`), `tests/test_object_vs_scene.py`; `figures/fig_object_vs_scene.png`. CPU only, Mac, 16:27–16:45 ET.
[^geo]: `results/p2_geodesic_direction_L{12,22}.json` and `results/GEODESIC_NOTES.md` (path metrics per arm: min readout radius, A.7 E_BC, excess nearest-real distance, L_G under each energy, in-plane share and cosine of the bend, closest-point distances; `targets_run` is empty in the lite files and the shards live outside the repo; no git commit recorded); `scripts/run_geodesic.py` (`length`, `g_sqrt`), `tests/test_geodesic.py`. CPU on box 53235298, 16:05–16:33 ET.
