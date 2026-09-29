# V-JEPA 2 physics take-home: report

Every number below comes from a file in `results/` (including `results/split70`, `results/split70_p2`,
`results/arcs`, `results/stimuli`) or `artifacts/gpu_session1.json`, named in a footnote or in the table's source line.
Paper claims are stated qualitatively and attributed to the paper.

## 1. Summary

The README asks for a small-scale reproduction of Joseph et al. (arXiv 2602.07050) in the frozen V-JEPA 2 ViT-L/16
encoder, for direction, speed and acceleration, and an open-ended Goodfire spline-steering extension (arXiv 2605.05115).

Sample sizes, units of resampling and the smallest effect each design could detect are tabulated in §7.1. **Three defended claims.** (1) The per-patch code is training-selective: on the supplied clips V-JEPA 2's mean
per-position R² is 0.96 at block 6, while the random-init network pools to 0.86–0.88 at every depth but its mean
per-position R² never exceeds 0.39; on a harder rendered set the half-frame jump replicates at the paper's depth on all
three seeds, a shape the random-init network never shows. (2) Every variable needs tens of probes at the paper's layer,
far outside a random-removal band, and Step 3 reproduces in shape and, with C.11's Adam probe sequence, the basis C.12
steers along, in count: it takes 18 to reach 10°, where the paper reports ≈ 12° at about 20. (3) Direction lies on a
ring, and at held-out direction values the spline path stays on the ring while the straight path cuts across the hollow
in the ring plane (in the 64-D subspace it is as close to real clips at point 12 and closer at point 22):
the minimum readout radius is higher for the spline on every arc (point 22 on the labels angle[^src]).

**Two disagreements with the paper.** (1) On the harder set transfer does not appear only after the Physics Emergence Zone (blocks 8–9): it is 0.7 at
point 1, and the per-position curve rises most between points 4 and 6 on every seed, though under the 90%-of-max rule
its onset is 9 / 8 / 8, her depth, and the sigmoid inflection 5.7–5.9 (§3.1): partial agreement on a harder stimulus,
not on the supplied one. What the zone is on that set: a heading code that transfers across the two halves of the frame at point 1 (half-frame R² 0.71, mean of three seeds) stops transferring through points 4–8 (−1.43 at point 8) and transfers again from point 9 (0.58 at point 12), so the sequence is shared → position-specific → shared, not the local → global transition C.5 (l.994–995) reads from its own data; the untrained copy never transfers[^seeds]. Physics is readable before the zone; what the zone marks on this stimulus is where a shared code is rebuilt. (2) Whether speed needs
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
the learned probe buys is specificity, not target error. Across model size (ViT-L / ViT-H / ViT-g) direction becomes readable in the first tenth of depth (onset 0.083 / 0.094 / 0.10 of depth) while untrained copies read 0.85–0.87 from block 1, so the early onset is mostly architecture and training lifts the ceiling to 0.99; on the hard render (ViT-H only, seed 0) the half-frame transfer first recovers at absolute block 9 in both sizes (0.28 of depth for ViT-H against 0.375) but recovers durably only at block 14 for ViT-H (0.44), so the direction of the shift depends on the rule (§3.5). Almost every head puts high attention density on the disk's tokens, yet at the ablated blocks the eight with the highest previous-slot density matter no more than random heads of the same number (0.040 against 0.12–0.22 of point-12 direction R²), a null that neither tests nor contradicts the paper's local-attention masking result (§4.7).

**One structural finding.** After the Physics Emergence Zone the direction code is better described in polar than in Cartesian coordinates at matched rank, as one harmonic (cos θ, sin θ) plus speed; the untrained copy is Cartesian from point 1 on (§4.4)[^coord]. This is an encoding result: in steering on constant-velocity carriers at the chord's norm Cartesian edits beat polar 2-D edits. That the ring is band-limited to k ≤ 2 comes from the centroid DFT (§3.2), not from this competition, which does not show the second harmonic is needed. In a rank-matched competition scored on held-out clips, the rank-2 head-to-head polar (cos θ, sin θ) minus Cartesian (v cos θ, v sin θ) is +0.015 [0.009, 0.022] held-out R² at point 12 and +0.009 [−0.001, 0.018] at point 22, ties at points 1–8 and excludes zero from point 9 on (except points 11 and 22); the untrained copy prefers Cartesian from point 1 on (−0.009 to −0.020; a tie at point 0). No candidate reaches the matched-rank knot-PC ceiling (margins −0.07 to −0.38), so this is a comparison between hand-built frames, not "the model's own coordinates". An earlier search that picked a 5-feature polar-Fourier frame at 6 of 7 points was a rank effect. A straight edit in (cos θ, sin θ, cos 2θ, sin 2θ) still crosses the ring's hollow like the chord (minimum path radius 0.61 / 0.63); through the predictor it is slightly worse than the chord frame-wide at block 22 (28.6° against 25.7°; paired Fourier − chord +2.9° [1.0, 4.5] at natural size, +4.0° [2.3, 5.6] at own size; the paper's spline 12.3°) and best on block-12 disk tokens at the natural norm (33.0° against 40.8°; at own norm every disk-token arm stays at the unedited level, 89–103° against 92.1°). On a rendered grid where acceleration is decorrelated from mean speed, signed acceleration beyond mean speed and displacement appears in the averaged features at the zone (R² 0.37 at block 8, 0.44 at 12; about 0 at blocks 1–4, a tight null; untrained ≈ 0) and in time-ordered per-tubelet features from block 1 (0.53); the paper's Cartesian (ax, ay) target is about 0 in averaged features through block 12 (0.17 / 0.46 at blocks 16 / 22); and the magnitude |a| is not readable before the zone, becomes readable through the signed code at the zone (0.22 / 0.44 at blocks 8 / 9, per-tubelet) and is weak by a direct linear probe late (0.12–0.16 at blocks 16–22), not distinguishable from zero under a cell-level bootstrap. On the supplied set, where every clip starts at rest, acceleration, mean speed and displacement are identical by construction. Whether it is a direct code or the per-step speed sequence is not decided: acceleration is an exact linear function of the per-step speeds, so removing in-set decoded speeds zeroes it by construction, while a speed decoder fit on other clips leaves 0.32 / 0.46 (points 12 / 22), room for a direct code; the paper's single-MLP claim cannot be tested as designed (§5).

**One new thing.** Frame-wide edits at points ≤ 12 wash out within a few blocks and barely reach the predictor's forecast, though the same edit placed on the disk's tokens at block 12 does reach it (40.8° from target on the disk-plus-twin set, 44.7° on the source disk alone, 33.0° for the four-number edit, against 92.1° unedited; §4.5); point-22
edits survive to the output, and along the point-22 path the forecast's heading code follows the intermediate
directions along the spline and jumps along the chord (−13.4° paired, −31.3° at large shifts). This causal headline is
about the heading readout of the forecast, not the whole forecast: without the heading probe every block-22 edit leaves
the forecast nearer the source's forecast than the twin's (forced choice ≤ 0.12), moving it a fifth to a quarter of the
way (whole-forecast recovery 0.15–0.26; twin identification 0.31–0.33 against chance 0.25 and 0.92 for the real twin),
and probe-free the spline's lead over the chord is mixed: at own norm it leads on all three probe-free readouts
(forced choice 0.045 against 0.016, recovery 0.172 against 0.154, twin identification 0.331 against 0.310), while at the
natural norm recovery favours the chord (0.186 against 0.234) and twin identification the spline (0.331 against 0.312);
the own-size lead is confounded by the spline's larger edit (about 1.4×; ratio of medians 1.37): at matched size (both at the twin's norm) the chord leads on forced choice (0.094 against 0.059) and recovery (0.234 against 0.186) while the spline keeps twin identification (0.331 against 0.312, +0.019 [+0.001, +0.037]) and the heading-probe lead (12.3° against 25.7° at that size; §4.5). This is a probe of the forecast on one
stimulus and one 45° arc, not a rendered future. The wash-out holds at the edits' own norms, and speed shows it too. Scaled to the
natural twin change, though, a point-12 speed edit moves the forecast's speed readout 0.57 of the way to the target,
so for speed it is partly a matter of dose. Direction is not: at the natural norm a point-12 direction edit leaves the forecast 79.7° [72.5, 86.6] from the target with the chord and 87.9° with the spline (label-free knot order), against 92.1° unedited and 12.3° for the point-22 spline[^natdir]. Through point 12 the direction the forecast ends up carrying originates in the disk's tokens (swapping them moves it 0.88–0.99 of the way, §4.6; this is routing through the remaining encoder blocks, not a predictor-specific read). A frame-wide edit is repaired in proportion to its size (a random edit of the same norm decays faster still), and the same edit placed on the disk's tokens alone at the twin's per-token dose gets partway through (40.8° from target, R 0.62), where the same edit on as many background tokens, or at matched energy on all of them, does not (88.9°, 80.1°; §4.5); on the source disk's tokens alone, without the target twin's disk location but still with the twin's per-token dose, it reaches 44.7° (65.8° with one uniform dose, 49.6° with a twin-free dose profile of the same total size) (the union set's extra tokens do nothing on their own), and the (cos θ, sin θ, cos 2θ, sin 2θ) edit there reaches 37.0°, while block 12 beats blocks 8, 16 and 19 (86.0°, 69.1°, 76.5°). A bounce edit written into the encoder turns the forecast toward the reflected heading with the same depth-and-token pattern (0.45 on the disk's tokens at block 12, 0.48 on all tokens at block 22), but a no-wall heading edit of the same norm does as well or better (0.46; 0.77), so what is written is the post-contact heading, not a contact event (§4.5).

**One negative.** At held-out endpoints the paper's comparison baseline, the chord between the raw centroids (A.9), run
with our matched-support edit, lands closer than our smoothing spline: over 16 held-out arcs on one arc set, 6.65° against
8.13° at point 12 and 4.20° against 6.46° at point 22 (paired +1.48° [1.27, 1.69] and +2.26° [2.09, 2.42], clip bootstrap
within arc; §4.3 unified bake-off)[^bake16]. On the headline arc alone the gap was +5.0° [3.9, 6.1] and +7.0° [6.0, 8.1]
(4.7° against 9.7°, 3.6° against 10.7°), the extreme case at point 12 and second-largest at point 22 (one arc, seed 16, reaches +8.3°); in the earlier per-arc runs the raw chord's lead across the 16 arcs is +1.3° ± 1.8 SD at point 12, with the spline ahead on 3,
and +2.3° ± 2.0 at point 22, with the spline ahead on none. The spline ties only the chord between its own smoothed knots, which was our line arm until
the parity audit[^rawchord]. A diagnosis run after the fact (§4.3) traces most of the point-22 endpoint loss to our own FITPACK smoothing spline: with the paper's interpolating spline the chord's lead over 16 arcs falls from +2.33° to +0.84° (spline ahead on 4/16). That story holds at point 22 only: at point 12 on value-ordered knots the interpolating spline is worse than our smoother (+0.81° → +1.56°, ahead on 2/16), and it loses nearest-real R to the chord on every arc at both points. A cross-validated Reinsch smoother, a rule written after seeing the headline arc and scored on the test read (post hoc, exploratory), wins by under 1°, a margin the true held-out centroid itself does not reach on the same probe (it trails the chord by 0.3°), so that win is reader alignment, not a better aim point. On one arc set, every curved arm keeps the higher path radius (minimum radius +0.21 to +0.33 over the chord at both points), while the paper's interpolating spline on our stored label-free point-12 angle overshoots off the ring (endpoint 34.5°, mean radius 1.52) and at point 22 owes part of its radius to a 1.43× larger edit; a straight edit in polar-harmonic coordinates (cos θ, sin θ, cos 2θ, sin 2θ) lands 4.75° and 3.68° off at own norm, behind the probe-subspace steer at point 12 (4.12°) and behind the chord at the chord's norm at point 22 (+2.13° [1.94, 2.32]), with the chord's path radius (§4.3). Two statements hold side by side and should not be merged into "curves buy the path, not the endpoint": at the encoder's held-out endpoint the spline does not beat the chord, while through the predictor at point 22 the heading code of the spline's forecast is much closer to the target than the chord's (11.3° [10.4, 12.3] against 27.2° [24.4, 30.5] at own norm and 12.3° [11.2, 13.6] against 25.7° [23.2, 28.3] at the natural norm, 200 carriers × 4 targets; §4.5), with the caveat that the own-norm spline edit is larger (§4.5). Speed and acceleration are straight, and there the spline
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
  background with a fixed camera. Acceleration clips all start at rest (`speed_mps` = 0.0 in all 1,536 metadata files),
  so acceleration, mean speed and displacement are identical by construction in the supplied set (mean speed = 0.3125 s
  × a and displacement = 0.195 s² × a; measured from the pixels, r = 0.9994 with mean speed). An independent audit from
  the pixels finds the block-1 ridge R² for acceleration, 0.977 ± 0.002, falls to −0.006 ± 0.008 once pixel-measured mean
  speed and displacement are partialled out (untrained 0.913 → −0.004), and a speed probe fit on the constant-speed set
  scores 0.975 on the acceleration clips (slope 0.299 s against the kinematic 0.3125 s)[^aaudit]. The paper's
  acceleration set has the same identity (l.681–684, "initialized at rest"). A rendered grid resolves it
  after the fact (240 clips, 4 mean speeds × 5 accelerations including decelerations, correlation of acceleration with
  mean speed 0.000; §5): there the introduction's claim (l.121–123, echoed by Table 1) that acceleration "can be
  approximated directly by a single MLP" without a velocity intermediate becomes partly testable: signed acceleration is
  readable beyond mean speed, but whether through a direct code or the speed sequence is not decided by the partial-out
  test as designed[^adec]. The paper has two acceleration targets, and each of our results pairs with one. Table 1
  (l.78–80) cites §5.2, the Cartesian (ax, ay) result, "decodable with high R² from early layers" (l.266–269); scored
  directly on the grid[^acart] ((ax, ay) = a (cos θ, sin θ), mean speed and displacement partialled out), the averaged
  (mean-pooled) features read it at about 0 through block 12 (−0.01 to −0.00) and only late (0.17 [0.10, 0.23] / 0.46
  [0.37, 0.53] at blocks 16 / 22), per-tubelet features at 0.09 at block 8 (clip interval [0.00, 0.15], cell interval [−0.06, 0.18], leave-one-cell-out 0.01) and first clearing zero under the cell bootstrap at block 9 (0.165 [0.005, 0.291]), up to 0.62 at block 22, and the untrained copy
  about 0 everywhere, so §5.2's early claim does not hold on these clips. Signed tangential acceleration is a different,
  easier target (averaged features about 0 at blocks 1–4 and 0.37 from block 8; per-tubelet from block 1, 0.53). §5.3 (l.280–281) is the magnitude
  claim, "speed and acceleration magnitude are available from early layers": on the grid |a| is uncorrelated with mean
  speed, displacement and signed acceleration, is not linearly readable before the zone (≤ 0.07 under every readout at
  blocks 1–4), becomes readable through the signed code at the zone (0.22 / 0.44 at blocks 8 / 9, per-tubelet) and only
  weakly by a direct linear probe late (0.12–0.16 at blocks 16–22), so that claim does not hold on these clips either.
  On the paper's acceleration clips every clip starts at rest (l.684), so (ax, ay) is exactly proportional to the mean
  velocity (vx, vy) and |a| co-varies with mean speed and displacement, the likely source of its early-layer numbers; the
  paper does not say whether its |a| probe was fit on the acceleration set alone or pooled with the velocity set, and its
  own clips were not tested here[^amag].
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
| Attentive-MLP probes | §3.2: patch-preserving attentive-MLP probes, "interpret both jointly" with the pooled ones; no recipe given | run after the fact at points 2 / 9 / 22 (direction), 1 / 9 / 19 (speed), 1 / 9 / 21 (acceleration), recipe ours (§3.1); the per-patch ridge probes of C.5 are the dense patch-preserving readout | ≤ 0.015 R² over the mean-pool ridge; points 3–8 not run, so the attentive onset is not located (§3.1) |
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

- Part 1, the Physics Emergence Zone: read per patch on the hard render across three seeds, the 90%-of-max onset is 9 / 8 / 8 (her depth) while the largest rise is at points 4 → 6 on every seed, and the half-frame dip-and-jump at points 8 → 9 (paper layers 7 → 8) is training-specific (§3.1).
- Part 1, Step 3: with our refit of the paper's C.11 Adam probe sequence as the steering basis the probe count reproduces, 18 probes to 10° and 16 to the paper's 12° at about 20 (§3.3).
- Part 1: a covariance-weighted edit built from one probe steers to 3–5°, outside the probe's plane, as does the same construction on a random 2-D subspace (median 4.2°), so the learned probe adds specificity, not reach (§3.3, §6).
- Part 2: the isometry verdict is set by the knot coordinate; under every label-free ordering point 22 is a tie (spline only under the basic interval in the fully faithful run), and Goodfire's own angle loses to the chord at point 12 (§4.4).
- Part 2: at the encoder output the verdict is mixed (the spline trails the chord by 10.8° at the chord's norm and leads by 2.1° at the natural norm), and in scalar extrapolation the spline trails the chord by 0.02–0.06 (§4.5, §4.3).
- Part 1, Step 2: the paper reads direction and speed off one velocity set and our stored counts came from two supplied sets; rerun on the same 750 constant-velocity clips under ridge, "speed needs fewer probes" holds in 7 of 8 cells under C.11's thresholds, which the paper's plotted Fig. 22 follows for speed (≈ 28 at layer 8) though not clearly for direction (Fig. 23's layer-8 direction sequence runs to about 100 probes against Fig. 22's 44), and in none under the caption's, so the verdict is a stop-rule call (§3.2).
- Part 2: our line arm joined the spline's smoothed knots; the paper's comparison baseline is the chord between raw centroids (A.9), which sits 1.9–3.7 PCA units from the spline point at the held-out targets and beats the spline at the endpoint by 5.0° and 7.0° on the headline arc and by 1.48° [1.27, 1.69] and 2.26° [2.09, 2.42] over 16 arcs at points 12 and 22 (unified run, §4.3), so "endpoints tie" became "the paper's chord wins the endpoint" (§4.3, §4.4).
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
at block 1 / point 8 / point 9, so no transition shows in the pooled readout. On the supplied set acceleration is mean speed in disguise (every clip starts at rest); on the decorrelated rendered grid (§5) the mean-pooled vector reads acceleration beyond mean speed and displacement only from point 8 (R² 0.37 at point 8, 0.44 [0.33, 0.53] at point 12, 0.79 at point 22; about 0 at points 1–4; untrained about −0.01), and partialling out the in-set decoded per-step speed sequence removes it, which it would by construction for a direct code too (§5). That is signed acceleration, and it appears in the averaged features at the zone, not after it; time-ordered per-tubelet features read it from block 1 (0.53 [0.46, 0.60]), as a per-step speed code allows, while the order-blind mean-pooled vector reads it only from block 8, so a pooled code does see the sign past block 4. The magnitude |a| (the paper's §5.3 target) is not readable on the grid before the zone (≤ 0.07 under every readout at blocks 1–4), becomes readable through the signed code at the zone (calibrated magnitude of the per-tubelet signed prediction 0.22 / 0.44 / 0.52 at blocks 8 / 9 / 12) and only weakly by a direct linear probe late (mean-pooled 0.12 [0.01, 0.21] / 0.16 [0.04, 0.26] at blocks 16 / 22), and the paper's Cartesian (ax, ay) target itself is about 0 in averaged features through block 12 (0.46 at block 22)[^acart], so neither the paper's early Cartesian (§5.2) nor its early magnitude (§5.3) acceleration result reproduces on decorrelated clips with pooled features[^amag]. Disk-pooling changes little
(direction peak 0.994, onset still 2[^disk]). The onset does not depend on how the CV folds are grouped (the paper's
App. B says "5-fold grouped" without the key): direction onset is 2 [2, 2] with stratified, direction-grouped, start-grouped and
8-sector-grouped folds (block 1 fold-mean R² 0.875 / 0.847 / 0.869; the sector figure, 0.828, is pooled out-of-fold R²,
since a one-sector fold makes per-fold R² meaningless), and speed onset 1 [1, 1] with stratified and
speed-grouped folds[^gcv]. Nor does it depend on the probe recipe: App. B's 20-config lr × weight-decay sweep (App. B names no optimiser; we used Adam, as C.11 does) at points 0–10 puts
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
selective (the random network stays flat). This does not explain the missing zone:
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
(0.881 → 0.861). On the pooled curve this is a partial recovery of the zone (point 5 is about a fifth of the
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
the per-position code from point 6 on (supplied: V-JEPA 2 0.94–0.98 against 0.22–0.39 for the random network;
hard: 0.80–0.97 against 0.57–0.62 on seed 0 and 0.52–0.61 on seed 1), not pooled availability; the half-frame dip
and jump are training-specific too (the random network's cross-half has no dip on either seed), though its level at
point 9 is not.

**Objective axis: VideoMAE** (v1 ViT-L, `MCG-NJU/videomae-large`, pixel reconstruction; 224-px input, 1,568 tokens; the paper used the VideoMAE-v2 family)[^obj]. VideoMAE matches V-JEPA 2
on every variable. Direction: block 1 0.886 vs 0.875, peak 0.992 (point 21) vs 0.991 (22), onset 2 for both. Speed:
peak 0.996 vs 0.994, onset 1. Acceleration: peak 0.996 vs 0.992, onset 1. Nested K at each model's peak is 67 / 109 /
87 for VideoMAE vs 88 / 89 / 67 for V-JEPA 2. Steering reaches the bar with 4 / 8 / 9 probes vs 4 / 7 / 6
(`figures/fig5_objective_axis.png`). Nothing in the pooled Part 1 measures on this stimulus is specific to latent
prediction. Per patch, run after the fact[^vmpp]: VideoMAE's per-position direction code matches V-JEPA 2's from point 8 on (mean per-position R² 0.961 / 0.972 / 0.981 at points 8 / 12 / 22 against 0.958 / 0.975 / 0.946 for V-JEPA 2 and 0.27–0.39 for the untrained copy) but arrives later (0.817 at point 6 against 0.957; per-position onset 8 [8, 8] against 5 [5, 5], an upper bound for VideoMAE, whose sampled points skip 3 and 7, so the lag is two to three blocks). Its half-frame transfer on the supplied clips is strongly negative through point 6 (−24.1 at point 1, −7.2 at point 6), where V-JEPA 2's never is (0.82 at point 1), and first positive at point 8 (0.59; 0.05 at point 9); that is not the V-JEPA 2 hard-render dip, which is a different stimulus and starts from positive transfer, and VideoMAE's absolute position embedding is a likely cause of position-specific early codes. So the per-patch code is what training adds beyond an untrained copy, and it is not specific to V-JEPA 2's objective among the two objectives compared: pixel-reconstruction training builds the same per-position code, two to three blocks later. The two models also differ in input (224 against 256 px; 1,568 against 2,048 tokens), pretraining data and position encoding, the confounds the paper itself lists (l.240–243), so the lag is not attributable to the objective alone. Caveat: VideoMAE adds an absolute sinusoidal position embedding and V-JEPA 2 uses RoPE, so the half-frame and pooled-patch comparisons mix training with position encoding; the per-position probes are unaffected.

**Attentive probe (extension; the paper reports attentive-MLP results on IntPhys only)**[^att]. Four learned queries
with a softmax over the 256 spatial tokens (each averaged over the 8 time steps, so temporal structure is not available
to it), a small MLP head, AdamW over App. B's learning-rate grid and weight decay 0.01 / 0.1, chosen by 5-fold CV on the
ridge's train folds. At the paper's layer (point 9) and at the peak the attentive probe is at or slightly above the
mean-pool ridge (CV R²): direction 0.987 vs 0.980 and 0.995 vs 0.991, speed 0.993 vs 0.988 and 0.995 vs 0.994,
acceleration 0.990 vs 0.982 and 0.994 vs 0.992; direction CV error falls from 4.0° to 3.7° and from 3.0° to 2.3°. At
block 1 it is below the ridge for speed (0.976 vs 0.983) and acceleration (0.926 vs 0.977), and above it for direction
at point 2 (0.946 vs 0.931). So the patch-preserving readout adds at most 0.015 R² to a linear read of the pooled
vector and does not create a Physics Emergence Zone (0.946 at point 2); it was run only at points 2, 9 and 22 for direction (1, 9, 19 for speed; 1, 9, 21 for acceleration), so where its onset falls within points 3–8 is untested. The best weight decay was the largest swept (0.1) at every
point, so these numbers may sit slightly below what the probe can reach.

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
  Under ridge the lag-1 autocorrelation of per-round drops in R² is positive everywhere (0.42–0.95; a sawtooth gives
  negative values); under the paper's Adam recipe it is negative for both variables (direction −0.38 / −0.08 at point 8
  and −0.28 / +0.22 at point 9, batch 64 / full batch; speed −0.44 to −0.54), so neither recipe gives a
  direction-specific sawtooth. On within-15° accuracy it is positive under the nested protocol (0.26–0.78) and mixed under the paper
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
non-constant power from point 8 on (0.08–0.16 at points 2–7) and k ≥ 3 together 0.03–0.11. Whitened K = 1 is
guaranteed for any linearly readable 2-D target: once the target-residualised features are whitened, one two-output
least-squares probe and its rank-2 erasure remove all linear information (Belrose et al., arXiv 2306.03819), so all four
planted controls give 1 and the whitened count cannot tell a rank-2 code from a spread one. The planted controls
demonstrate that guarantee; they are not a calibration of it. What they do separate is in the other columns: shear
inflates literal K (8), a planted k = 3 harmonic shows up in the DFT (0.32) with literal K = 1, and copies do not
inflate literal K. So (i) direction is organised on a harmonic basis (k = 1 plus a cos 2θ term shared by opposite
directions), not on tens of independent axes (the DFT column), and (ii) the literal count behaves like Jin et al.'s
sheared circle: "tens of dimensions" reflects the covariance's shape, not an intrinsic rank. It does not follow that the
code is only a rank-2 plane: after the best rank-2 linear erasure (LEACE-2) an MLP still reads direction at 0.37 /
0.41 / 0.82 at points 9 / 12 / 22 (0.87 before erasure at point 22), so beyond the linear readout there is a nonlinear
residual that grows toward the output.

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
rises as MAE-to-target falls. (Sample sizes, resampling units and post-hoc detectable effects for this and every later steering result: §7.1.) **Figures:** `figures/fig3_steering_paper.png`, `figures/fig3c_steering_nulls_paper.png`,
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
  target (stored: 5, the ridge basis under the in-sample α = 100 judge; the paper's Adam basis needs 18); with 200 draws at every N[^n200] the rank-2K null is beaten at p < 0.05 from N = 4 for both
  (stored probe: N = 9; split-half probe: N = 11, sustained from 15), and the rank-matched null from N = 3 (stored: 5;
  split-half: 5). The
  null reading therefore depends on how the evaluation probe is regularised, which C.12 does not fix.
- **The paper's steering basis** (point 9, direction)[^adamb]. C.12 steers along C.11's Adam probe sequence, not a
  ridge one. Refitting that sequence (lr 1e-3, wd 1e-4, batch 64; it reproduces the stored Adam run and stops at K = 84)
  and steering with it: N = 1 / 2 / 3 / 5 / 10 / 20 / 84 give 84.1° / 81.1° / 77.7° / 69.2° / 27.9° / 7.3° / 2.9° to
  target (ridge basis: 78.4° / 61.1° / 25.4° / 8.7° / 3.1° / 2.9° / 2.7° at N = 37). Eighteen Adam probes reach 10°,
  against five ridge probes (the ridge basis judged by the α = 100 evaluation probe fit in-sample on the steered clips;
  the split-half judge gives 12.4° / 17.1° at N = 5, mean 14.7°; with 200 draws the five ridge probes' 8.7° is not
  distinguishable from the 74-dimensional random basis, median 11.8°, p 0.22, first beaten at N = 9, while the
  rank-matched null gives 54.9°, p 0.035) and the paper's about 20; at N = 5 the Adam basis is at
  69.2°, no better than the full-rank random basis (p 0.52); N = 1–5 give 84–69°, the paper's "modest improvement (MAE > 50)".
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

### 3.5 Model size: layer curves by fraction of depth

*Design*[^scale]. The same probes, targets and split as §3.1, on three V-JEPA 2 encoders, ViT-L (24 blocks), ViT-H (32)
and ViT-g (40), each trained and untrained (same config, random init, seed 0). The x-axis is block / L, with the patch
embedding at 0; curves are held-out test R², and onsets are the first point at ≥ 90% of the CV maximum (200-draw clip
bootstrap). ViT-H and ViT-g run in bf16 autocast; that precision was checked on ViT-L only (16 clips per set: median
relative L2 error 0.5%, rising to 7.2% / 9.3% at the last two read points; cosine ≥ 0.9957); the untrained ViT-H and ViT-g arms also ran in bf16, with no untrained parity
measured. The ViT-L arm re-probed the stored Part 1 fp32 features, so its match to the Part 1 files (within 2e-13 on every
variable and both copies) checks probe determinism, not the new extractor; the size comparison therefore mixes fp32 ViT-L
extraction with bf16 ViT-H / ViT-g, and the extractor's only evidence is the 16-clip parity per set.

*Results.* Direction becomes readable in the first tenth of depth at every size: onset block 2 for ViT-L (0.083 of
depth), block 3 for ViT-H (0.094), block 4 for ViT-g (0.10), each with a bootstrap interval of one point that reflects clip resampling only, and the CV peaks are
0.991 / 0.989 / 0.992. The onset depends on the threshold: at 95% of the maximum it is block 3 / 4 / 4 (0.125 / 0.125 / 0.10
of depth), at 85% block 1 / 2 / 3, so "early" holds at any threshold up to 95%. Where the peak falls is not stable: after about a third of depth every trained direction
curve is flat near 0.99 (CV 0.973–0.991 / 0.985–0.989 / 0.971–0.992), so the argmax wanders from 0.56 to 0.95 of depth
(blocks 22 / 18 / 38) and is not a size effect. The untrained copies read direction at 0.85–0.87 (CV maximum 0.868 / 0.856 / 0.846, reached by
block 1–2), so the early onset is mostly architecture, and training lifts the ceiling from about 0.86 to 0.99. The
untrained ViT-H and ViT-g curves decline late (0.767 / 0.743 CV at the last block against 0.865 for fp32 ViT-L); since they
ran in bf16 without untrained parity, and an untrained probe turned a 0.5% bf16 feature error into R² −15.7 in §4.7, that
decline is not read as a size effect. Speed and
Cartesian (vx, vy) are readable from block 1 at every size (onset block 1; peaks 0.989–0.995). The acceleration target on
this set is mean speed in disguise (§2: every clip starts at rest), so its curve (onset block 1 for ViT-L and ViT-H, block
4 [3, 4] for ViT-g; peaks 0.991–0.993) and the Cartesian (ax, ay) curve (onset block 1, peaks 0.987–0.989) repeat the speed
result rather than testing acceleration; the untrained ViT-g reads it from block 1 (peak 0.920 at block 1; (ax, ay) 0.975).
At the patch embedding (point 0) the trained ViT-H and ViT-g read direction at CV R² 0.709 and 0.718 (test 0.739 / 0.740)
against 0.10 for ViT-L (test 0.035) and 0.045 for untrained ViT-H (test 0.010), and the untrained ViT-H and ViT-g read
speed there at 0.380 and 0.382 where untrained ViT-L does not (−0.028). Point 0 is a Conv3d patch embedding whose pooled
output is its weight matrix times the position-averaged 1,536-d tubelet patch, so widths of 1,024 / 1,280 / 1,408 keep
different amounts of that patch; this is untested, and the untrained values, from bf16 arms, are not read as size
effects.

*The zone.* On the supplied clips there is no pooled zone to scale (ViT-L reads direction at 0.875 at block 1, §3.1), so
these curves are readability onsets, not a Physics Emergence Zone. The one zone-versus-size statement rests on the
hard-render per-patch half-frame readout (§3.1), run for ViT-H only (render seed 0; ViT-g and the untrained ViT-H not
run): the cross-half R² goes 0.61 / 0.37 / −1.02 / −1.64 / −2.44 at blocks 4–8 and back to 0.40 at block 9, as ViT-L's
does (negative at 4–8, 0.08 at 9), then −0.051 [−0.139, −0.014] at 10, 0.089 at 11, −0.332 [−0.476, −0.283] at 12 and
−1.627 at 13 before staying positive from block 14 (0.41). The direction of the shift depends on the rule: the first
recovery is at block 9 in both sizes, earlier as a fraction of depth for ViT-H (0.375 → 0.28), while the durable recovery
(positive from then on) is at block 9 for ViT-L (0.375) and block 14 for ViT-H (0.44), later. The JSON's
`zone_halfframe_hard.statement` uses the first-recovery rule. Steering by size (the ViT-H chord and spline through its own predictor) was not run. Compute:
4,572 GPU-s of forwards for ViT-H and ViT-g plus 92 s for the ViT-H per-patch extraction; the feature files stay on the
GPU boxes, with their sha256s recorded in the JSON.

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
  arc at each point. Over the 16 arcs of the unified run (§4.3) it trails the raw chord by +0.81° [0.74, 0.89] at point 22
  with a 1.43× edit, and on the stored label-free point-12 angle it fails (+27.9° [27.4, 28.3]). All steering uses the count-weighted smoothing spline, which I chose on train folds, because the interpolating
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
- **Speed and acceleration are straight.** The earlier "knot spacing is linear (R² 0.999 vs 0.909 for log)" used the
  cumulative chord length between consecutive knots, which scores 0.999 with shuffled labels too, so it says nothing
  about spacing; a label-free coordinate (the centroids' first PC) slightly prefers log speed (log − linear R² +0.061
  [0.033, 0.084] at point 12, +0.090 [0.071, 0.109] at point 22; acceleration shows no clear preference)[^sacc]. Centroid PR is 1.65 / 1.67 for speed (points 12 / 19) and 1.61 / 1.66 for acceleration (12 / 21). The line
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

**Direction, contiguous 45° arc** (the headline arc, seed 0: one arc; the 16-arc means are in the next two tables and
in the unified bake-off at the end of this section, raw-chord endpoint gap +1.48° [1.27, 1.69] / +2.26° [2.09, 2.42] at
points 12 / 22 over 16 arcs; manifold = spline, linear = chord; gaps are manifold − linear with a 95% paired clip
bootstrap):

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

**Why the spline lost the held-out endpoint (diagnosis after the fact)**[^epd]. At point 22 most of the loss was our
spline, not the method; at point 12 it was not. We smoothed each PCA coordinate separately with FITPACK (`splrep`, s = m, its own knot subset), which
neither the paper (an interpolating periodic cubic, A.3) nor its code (causalab's Reinsch smoother, one λ for all
coordinates) does. That spline reads 3.4° off at its own knots against 1.9° for the raw centroids at point 22, its
aim point reads 4.3° off against 1.5° for the chord's, and our additive edit counts that error twice, once at the
target and once at the source anchor (the source term alone costs 1.0–1.7° over the 16 arcs). Run on the same knots,
the authors' own code (identical to our interpolating spline to 1e-13; cosine 1.0 with our edit vector) turns the
point-22 headline arc from +7.05° to −0.48° (spline minus raw chord), but over the 16 arcs the interpolating spline
still trails the chord (+0.84°, spline ahead on 4/16 at point 22, down from +2.33° for our smoother; +1.56°, ahead on
2/16 at point 12 on value-ordered knots, which is worse than our smoother's +0.81°; +27.2° ± 43.2 on the label-free
knot order used for the stored point-12 results) and loses nearest-real R on every arc, because across a 45° gap the interpolant overshoots to 12 PCA units
from the true held-out centroids against 5 for the chord. Norm is the second cause, and it cannot be separated from the
source anchor: rescaled per clip to the stored spline's norm, the raw chord reads 6.64° against the spline's 6.57° over
the 16 arcs at point 22 (spline minus rescaled chord −0.07°, spline ahead on 11/16; −0.16°, 10/16 at point 12 on
value-ordered knots), so at the spline's norm the two arms are within 0.2°, though that per-clip rescaling was not a
comparison fixed in advance either; on the headline arc norm accounts for 4.34° of the
7.05°, and over the 16 arcs the rescaling closes the +2.33° gap to −0.07°. The reverse match, the spline at the chord's per-clip norm, reads 7.75° against the chord's 4.24°, so per-clip
rescaling costs each arm 1.2–2.4°[^epdnorm]. The `aim="arc"` option of 42b30fe makes the spline
worse (+3.2°); the real fix for the point-12 knot problem is value-ordered knots (+1.32° → +0.81°). With Reinsch
smoothing and λ chosen by leave-block-out cross-validation on the kept knots alone, the spline beats the chord on all
16 arcs at both points (scored on the test read after the rule was written; post hoc, exploratory), by 0.45° ± 0.21 (point 22) and 0.85° ± 0.34 (point 12, value-ordered), and on the MLP and
nearest-real readers too, a margin the true held-out centroid itself does not reach (steered to it, the probe trails the
chord by 0.32°, better on 4/16 arcs at point 22; 0.61°, 4/16 at point 12), so the smoother's win is reader alignment of
denoised knots, not a better aim point, and it is exploratory; that rule is not in the paper, it was written after seeing the λ sweep on the headline arc
(before scoring the 16 arcs), and about half of the gain is denoised knots rather than curvature (a chord through the
smoothed knots gets −0.21° and −0.52°). The ring's bend across the gap is real (the true held-out centroids sit 0.78 /
0.90 of the way from the chord to the smoothed curve) but smaller than the held-out centroids' own noise (2.0 / 1.4
against 3.9 / 2.6 PCA units), and the ridge probe reads chord points almost exactly by construction (the chord's aim
point reads 1.5° off, the true held-out centroid 2.9°), which is why the oracle aim above does not beat the chord. So at this gap width the endpoint cannot separate the two methods by more than about 1°, the earlier
"chord beats spline" numbers in this section overstate the method's loss by 5–7° on the headline arc and 1–2° over 16
arcs, and the spline's advantage is on the path (readout radius, ordering, and the forecast's heading code following it in §4.5, a probe readout),
not at the endpoint. The predictor-level results of §4.5 used the interpolating spline and are unaffected.

**All six arms on one arc set (unified bake-off)**[^bake16]. The comparisons above were run piecemeal (different arms on
different arc sets, some on the headline arc only). The unified run puts every arm on the same 16 contiguous 45° arcs at
both points (point 12 on the stored label-free angle, point 22 on the labels angle), 8 held-out targets × 48 test
clips per arc, an 11-waypoint walk, each arm at its own norm and rescaled per clip and per waypoint to the raw chord's
norm; intervals are a clip bootstrap within arc (all targets of a clip together, 1000 draws) on the mean over arcs of
per-arc means. It reproduces the stored FITPACK headline path radius exactly (min radius 0.8616 against 0.6289 for the
raw chord at K = 50 on the headline arc, stored and recomputed identical to 1e-15; 0.865 / 0.632 at K = 11).

| arm (own norm) | pt 12 endpoint err | pt 12 R | pt 12 radius min / mean | pt 22 endpoint err | pt 22 R | pt 22 radius min / mean |
|---|---:|---:|---:|---:|---:|---:|
| raw chord (A.9) | 6.65° [6.49, 6.82] | 0.202 | 0.59 / 0.76 | 4.20° [4.09, 4.32] | 0.181 | 0.61 / 0.78 |
| paper's interpolating spline (causalab, smoothness 0); pt 12 on our label-free angle, see (i) | 34.5° [34.0, 35.0]; median over arcs 14.7° | −3.15 | 0.80 / 1.52 | 5.02° [4.90, 5.14] | −0.027 | 0.92 / 1.00 |
| knot-CV Reinsch smoother (exploratory) | 6.27° [6.12, 6.43] | 0.222 | 0.91 / 1.00 | 3.76° [3.65, 3.85] | 0.206 | 0.94 / 1.00 |
| our FITPACK smoother | 8.13° [7.92, 8.33] | 0.182 | 0.86 / 1.02 | 6.46° [6.30, 6.64] | 0.200 | 0.88 / 1.00 |
| straight edit in (cos θ, sin θ, cos 2θ, sin 2θ) | 4.75° [4.62, 4.88] | 0.225 | 0.61 / 0.78 | 3.68° [3.57, 3.78] | 0.201 | 0.63 / 0.80 |
| probe-subspace steer (our refit; see below) | 4.12° [4.02, 4.23] | 0.143 | 0.62 / 0.80 | 4.24° [4.13, 4.35] | −0.153 | 0.60 / 0.78 |

Paired against the raw chord (arm − chord, own norm): endpoint +27.9° [27.4, 28.3] (interpolating), −0.38°
[−0.50, −0.27] (Reinsch), +1.48° [1.27, 1.69] (FITPACK), −1.90° [−2.04, −1.75] (Fourier), −2.53° [−2.72, −2.34]
(probe) at point 12; +0.81° [0.74, 0.89], −0.45° [−0.50, −0.40], +2.26° [2.09, 2.42], −0.53° [−0.62, −0.45] and
+0.04° [−0.11, 0.18] at point 22. Minimum path radius: +0.21, +0.32, +0.27, +0.02, +0.03 at point 12 and +0.31,
+0.33, +0.28, +0.02, −0.01 at point 22, every interval excluding zero except the probe arm's mean radius at 22.
Monotone fraction: the three spline arms 0.84 / 1.00 / 1.00 at point 12 and ≥ 0.999 at point 22, the three straight
arms 0.96–0.98. This resolves the path-radius caveat: the path advantage is a property of every curved arm, not of our
FITPACK smoother alone, and the straight edit in polar-harmonic coordinates, which is among the best endpoint arms at own
norm (4.75° against 4.12° for the probe-subspace steer at point 12; 3.68°, best, at point 22) but loses to the chord at the
chord's norm at point 22 (+2.13° [1.94, 2.32]), has the chord's radius profile (it dips to 0.61–0.63 mid-path like the chord). Three qualifications. (i) At
point 12 on the stored label-free angle the paper's interpolating spline fails: its endpoint is 34.5° off, its mean
radius 1.52 and its waypoint radius climbs from 0.99 to 2.32 along the path, so it leaves the ring outward, with an edit
6.5× the chord's norm, because the knots crowd where the unsupervised angle compresses; the endpoint diagnosis gives the
same (33.9° on that configuration) while on value-ordered knots the same spline is within 1.6° of the chord
(`configs.L12_unsup` and `L12_labels`[^epd]). Its high minimum radius at point 12 is therefore overshoot, not ring
fidelity. The 34.5° mean is carried by three arcs (131.9°, 129.1° and 94.7°): per-arc means run 5.5–131.9° (median
14.7°), 9 of 16 arcs keep a mean radius ≤ 1.02 (on the ring) and one reaches 5.24, and the arc-level half-width of its gap
to the chord is about ±21° (SD 42.9 over 16 arcs), so the clip-bootstrap interval printed in the table is far too
narrow for this arm. The QA audit's recomputation of the knots (not stored in a results file) finds a minimum knot
spacing of 0.168° (arc 0) and 0.015° (arc 5) on the label-free angle against 5.625° on the labels, and 7 of 56
consecutive knot steps out of value order; given the label value as its coordinate, which is our departure (the paper's cyclic coordinate is a label-free atan2 of
the top two PCs, A.3, and causalab's cyclic configs use `intrinsic_mode: pca`), the same spline is +1.56° from the chord.
Knot order also accounts for 32.2° of the 55.7° gap between the spline and the chord as a disk-token edit through the
predictor (100.4° on label-free knots against 68.1° on label-ordered knots, chord 44.7°; §4.5); 23.5° remain. (ii) At point 22 its radius advantage is partly dose: its edit is 1.43× the chord's norm, and at the chord's
norm its minimum radius falls from 0.92 to 0.74 and its endpoint error rises from 5.0° to 17.0° [16.7, 17.5]; the
Reinsch and FITPACK arms keep 0.86 and 0.84 at the chord's norm (endpoint 5.0° and 7.6°), so the dose-clean path radius of a curved arm at point 22 is 0.84–0.86 against the chord's 0.61, not the interpolating spline's 0.92. (iii) The probe-subspace steer at
point 22 uses a 1.5× larger edit than the chord (37–56 probes per arc); at the chord's norm it reads 22.8° [22.0, 23.7]
off (R 0.000), so its tie with the chord at its own norm is bought with dose. At point 12 (16–22 probes) its −2.53° lead
becomes a +0.27° [0.03, 0.49] deficit at equal norm, and the Fourier edit's lead shrinks to −0.34° [−0.51, −0.18]. The
Reinsch λ chosen by knot-block CV was 1e-4 on all 16 point-12 arcs (1e-5 or 1e-4 at point 22). `figures/fig_p2_path_by_spline.png`
shows the waypoint radius along the path for each arm. Through the predictor the same four-number edit is 2.9° behind the chord frame-wide at block 22 and 7.8° ahead of it on block-12 disk tokens (§4.4). The probe-subspace arm is our variant of C.12, not the paper's
probes: a ridge sequence (not C.11's Adam), refit per arc on the 628–632 knot clips at kept values with the arc's 8
target values withheld, its length set by 3-fold nested CV (16–22 probes at block 12, 37–56 at block 22, against Part 1's
K = 64 / 88 on 1,200 clips), a unit (sin, cos) target, a single full-K solve with no N sweep, a straight walk, the
probe-fold reader rather than an evaluation probe fit on the test clips, and blocks 12 / 22 rather than 9. Its block-12
lead (−2.53° [−2.72, −2.34] at its own norm) turns into a +0.27° [0.03, 0.49] deficit at the chord's norm.

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
at held-out endpoints the paper's raw-centroid chord lands 1.5° / 2.3° closer than our smoothing spline over 16 arcs (5.0° / 7.0° on the headline arc), and it ties
only the chord between its own smoothed knots; the interpolating version rebuilds held-out centroids worse
than the smoothing one and edits 1.4–1.6× more than the chord (hence smoothing); on scalars it adds nothing inside the
knots and, as a smoothing spline, trails the chord by 0.02–0.06 in extrapolation even continued along its end tangent as
the authors' code does (their own interpolating arm is mixed: better on speed, worse on acceleration). **Failure cases**: the
held-out endpoint against the raw-centroid chord at both points over 16 arcs (+1.48° [1.27, 1.69], +2.26° [2.09, 2.42];
headline arc +5.0°, +7.0°) and on 13 of 16 point-12 arcs, and the position sheet.

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

**Conceptor arms over the 16 arcs, on correct knots**[^c16]. Over the 16 held-out arcs of §4.3, with the held-out aim
fixed at point 12 (#266: on seeds 1, 4, 5, 7, 8, 9, 11 and 13 the old aim built the conceptor's target condition
partly from knots of other directions), no conceptor arm steers direction at either point. The COAST gate at
COAST's own β = 0.3 ends 94.2° [92.6, 95.9] from the target at point 12 and 96.9° [95.2, 98.6] at point 22, against
about 93° unsteered; run out to β = 1 it ends 99.7° and 108.1° away, so the headline arc's 41.8° was a favourable
single case. The best conceptor arm, the target-aimed update rescaled to the chord's norm, ends 83.7° [81.7, 85.7]
and 75.7° [73.6, 77.7] away. Over the same arcs the raw chord ends 5.7° [5.5, 5.8] and 4.2° [4.1, 4.3] from the
target and the spline 10.5° [10.1, 10.9] and 6.6° [6.4, 6.7] (the point-12 spline mean includes seed 7 at 46.5°,
present in the stored fixed-aim run as well; under the legacy aim the 16-arc means were 8.0° and 6.7°), and no
conceptor arm beats the chord on any arc. COAST's aperture band is never met (α = 0.1 or 0.5), and only 2–33% of
each conceptor edit lies in the ring plane against 56–64% for the spline and chord. This uses the conceptor recipe
of the paragraph above; the COAST-faithful rerun follows.

**COAST as written, over 16 arcs**[^coastf]. I re-ran COAST as the paper specifies (Eq. 1–5, 7–11, App. A.9–A.10):
conceptors on the full 1024-d activation rather than the PCA-64 subspace of the run above, the paper's pseudoinverse
AND and Jaeger's singular-case AND, the Stage-2 overlap rule for α, COAST's gate h′ = h[(1 − β)I + βC_steer] with β in
COAST's grid, and COAST's own positive-only, linear (CAA) and random-eigenvector ablations, plus PCA-16/64 and
time-token variants. On the headline arc the faithful gate (Jaeger AND, β = 0.3) reads 88.1° / 90.3° at the raw
chord's norm at points 12 / 22 (unsteered 88.9° / 88.7°, chord 4.7° / 3.6°, COAST's linear baseline 5.0° / 3.7°), and
over the 16 arcs every conceptor arm in the full and PCA-16 spaces reads 91.0–102.0° at the chord's norm, worse
than the chord (6.7° / 4.2°) with a CI above zero on 16/16 arcs. The gate equals its random-eigenvector control (88.0°
against 88.1° at point 12, 89.4° against 90.3° at point 22), because C_target AND NOT C_source keeps a trace of 0.72 /
5.09 of 1024 and the gate reduces to a near-uniform shrink h → (1 − β)h. The mechanism is the one stated above: a
conceptor is a soft projector onto a condition's centred covariance, so it sees the shape of each value's cloud but not
where the cloud sits, and on the ring neighbouring values are shifted copies of nearly the same cloud (source–target
overlap 0.27–0.61 in the full space, never in COAST's [0.85, 0.95] band); a PSD gate with β ≤ 0.3 also cannot rotate a
centred state by more than about 10°, while 46% of the steers need more than 90°. The only conceptor arm that reaches
the target is not COAST: Jaeger's uncentred conceptor of the target clips alone at β = 1 (outside COAST's grid), which
projects onto the span of the target clips, mean included. At its own norm, about 1.8× the chord's, it beats the chord
over 16 arcs at point 12 (2.5° against 6.7°, CI below zero on 16/16) and ties it at point 22 (4.3° against 4.2°); at
the chord's norm it reads 15.1° / 14.1°, worse on 16/16. COAST's linear baseline is a difference of means and ties the
chord (5.5° / 4.1° over 16 arcs). So conceptor steering of a ring code fails here as a method, not through a recipe
error; the predictor-side version (gate the encoder, score the forecast) was not run.

**Which probe directions does the predictor listen to (Makelov ranking)**[^mak]. Makelov et al. rank candidate feature
directions by their downstream causal effect rather than by probe accuracy. Applied to Part 1's INLP basis at point
22: each of the first 32 INLP directions was added to the context of the 16 headline-arc carriers at the norm of that
carrier's raw-chord edit, with both signs, and the predictor's forecast was read with the repair-analysis probe
(CV error 9.5°). Directions INLP found earlier move the decoded forecast angle more: Spearman ρ = −0.60 between INLP
order and mean absolute forecast move (carrier bootstrap [−0.64, −0.54], direction-permutation p = 0.0003), −0.77
over the 16 INLP rounds, and −0.38 over 64 directions. The first eight directions move the forecast by 31° on average
and directions 17–64 plateau at 11.6–14.3°, the level of random directions of the same norm (13.4°); only 28% of the 32
directions beat the random 95th percentile; six of the first 32 move the forecast further than the raw chord's 34.7° (up to 53.8°), none as consistently in sign. The overall
change in the forecast embedding is flat, about 0.08 relative for INLP, random and chord edits alike (ρ = 0.02), so
INLP order predicts where an edit lands in the forecast's direction code, not how much the forecast changes. The
per-direction effect is sign- and carrier-dependent (signed means under 25°, absolute means up to 54°), which is why
the ranking uses the absolute move. Point 12 was not run: no INLP basis exists there. This ranks individual directions of the ridge basis at point 22 under single, untargeted, fixed-norm edits; it does
not test the paper's coordinated, targeted N-probe steering at layer 8 on the Adam basis, and §3.3 shows the count
depends on that basis. What it says is narrower: early INLP directions individually move the predictor's forecast
more, and directions past about 16 are no better than random single edits of the same norm.

**Energy geodesic (Goodfire Eq. 4–6)**[^geo]. The paper defines the geodesic as the shortest path under
G_E(h) = (α e^{−E(h)} + β)^{−1} I (l.1396–1411) but never computes one, and causalab has no implementation, so
this is ours: 50 free nodes between pinned endpoints, Simpson quadrature, torch L-BFGS, two energies fit on knot clips
only (a kNN energy with the Levina–Bickel dimension, 13.1 at point 12 and 7.3 at point 22, and a whitened top-10-PC
KDE with cross-validated bandwidth), α, β calibrated as in Béthune et al. 2505.18230 §3.3 since the paper gives no
values. The full run covers all 8 held-out targets with 48 carriers each and a random-restart null (3 restarts on 8
clips per target). None of the 112 batched solves converged within 100 L-BFGS steps (a lite run on 3 targets had 1 of
42), so every path length is an upper bound and "geodesic" means a 100-step descent of Eq. 4; endpoints are pinned to
the raw chord's end state, so endpoint error and nearest-real R equal the chord's by construction. Started from the
chord, the descent stays near it: its bend has an in-plane share of 0.08–0.11 (spline 0.15 at point 12 and 0.41 at
point 22, a random direction 0.03) and a cosine with the spline's bend of 0.03–0.08. On the readouts it matches the
chord: minimum readout radius chord / spline / kNN geodesic / KDE geodesic 0.63 / 0.88 / 0.71 / 0.62 at point 12 and
0.63 / 0.85 / 0.65 / 0.63 at point 22; the kNN geodesic is 0.08 above the chord at point 12 (per-target CI above 0 in
8/8 targets) and ties it at point 22, and both geodesics sit 0.17–0.25 below the spline, CIs below 0 in every target.
A.7 E_BC is 1.52 / 0.88 / 1.28 / 1.31 at point 12 and 3.80 / 3.84 / 4.01 / 3.91 at point 22. The descent wins only on
what it minimises: excess distance to the nearest real clips +0.33 / +1.23 / −0.62 / +0.07 at point 12 and +0.64 /
+0.75 / −0.20 / +0.38 at point 22. Random restarts land 0.6–2.1 from the chord-started geodesic and 27–67% of them
reach an L_G more than 1% lower, so the chord-started path is not the global minimum; unconverged, this does not
separate a multimodal landscape from an unfinished descent, and the path shape depends on where the descent starts.
What does not depend on convergence is the fixed-path length: under G_E the spline is 1.8–2.0× longer than the chord at
point 22 (kNN and KDE energy) and 4.4–8.7× at point 12, so the density metric does not select the ring route. One
caveat: the point-12 spline arm in this run does not reproduce the stored raw-chord file (endpoint 14.9° against
9.7°; suspected scipy 1.15.3 on the box against 1.17.1 on the Mac in periodic `splrep`, unproven), so point-12
comparisons with the spline here are provisional; the chord arm reproduces the stored file at both points. Point 12
uses the label-free angle and point 22 the labels, as elsewhere.

**Subspaces, superposition and the coordinate competition**[^mg]. I fit one linear encoding model per point from
per-step activations to [cos θ, sin θ, cos 2θ, sin 2θ, speed or acceleration, start x, start y] plus the within-clip
step curve, and take each variable's coefficients as its subspace. In the noise-whitened metric (length is
signal-to-noise) the direction plane, its second harmonic, the speed axis, the start-position plane and the time
subspace are mutually orthogonal to within a random-subspace null at points 8–22 (every overlap ≤ 0.034 against a null
p95 of 0.034–0.038). The one shared direction is speed with acceleration (overlap 0.94 / 0.89 / 0.79 at points 1 / 12 /
22, and 0.89–0.94 in the untrained copy, so this part is stimulus), and the ten coded dimensions have a participation
ratio of 8.4 at point 12. Steering agrees: with in-subspace edits matched in norm, every off-diagonal leak is at most
0.67 of the readout's natural spread except speed and acceleration reading each other (1.36–1.78×), and across the 80
cells leakage follows overlap with the readout probe's weights (Spearman 0.72, p 6e-14) more than subspace overlap
(0.42). The direction edits' speed leak is a property of how centroid edits are built: a ring edit inside the fitted
subspace leaks 0.09 / 0.04 spreads (points 12 / 22) against 1.46 / 1.76 for the raw chord, and 97% / 96% of the raw
chord's leak is explained by mean-speed and motion-mix imbalance between the clips behind its two centroids (Pearson
0.88 with speed imbalance). Projecting the ring edit orthogonally off the speed subspace raises its leak to 1.50×; an
oblique projector along the speed encoding direction removes it (on the independent MLP reader the raw chord's leak
falls from 1.07 to 0.83 spreads, −0.24 [−0.31, −0.17], direction cost −0.10°, point 12; −0.07 [−0.14, 0.01] at point
22). *Coordinates.* The first analysis here (`exp4` of the motion-geometry run) picked (cos θ, sin θ, cos 2θ, sin 2θ, v)
at 6 of 7 points on transfer R². Its candidate list was written into the JSON at 17:10 ET, the code was first committed
at 17:47, and k = 2 was already known from the centroid DFT (§3.2), so it was not a blind registration; the winner had 5
features against 2–3 for the others with no matched-rank control, and its bootstrap intervals exclude their own point
estimates (0.763 against [0.661, 0.705] at one point), because each draw recomputes the cell centroids from resampled
clips, which adds centroid noise and biases the resampled R² downward. Its "polar" and "log-polar" candidates regress on
raw degrees (`src/wm/motiongeom.py` l.85–86: θ enters linearly, with a wrap discontinuity at 0°/360°), a strawman rather
than a polar arm. That search is superseded by a rank-matched competition[^coord] whose seven candidates (Cartesian,
polar 2-D, polar 4-D, polar 6-D, and each 2-D/4-D frame with v added), matched-rank knot-PC ceilings and random smooth
features of the same labels were written, with a script hash stored in the JSON, before its numbers were computed; it
covers all 26 points in both copies, fits on knot folds 0–2 and scores held-out R² on probe folds 3–4 (738 / 490
speed-set clips), with 500-draw probe-clip bootstrap intervals. Under the matched-rank rule polar 2-D + v wins at 24 of
25 trained points (points 2–25; Cartesian + v at point 1), but never reaches the matched-rank knot-PC ceiling (margins
−0.07 at point 9 to −0.38 at point 5), so a hand-built frame captures part, not all, of the rank-k signal. The
confound-free rank-2 head-to-head, polar minus Cartesian, ties at points 1–8 and is +0.014 [0.007, 0.020] at point 9,
+0.015 [0.009, 0.022] at point 12 and +0.009 [−0.001, 0.018] at point 22, with the interval above zero at every point
from 9 to 25 except 11 and 22, just after the Physics Emergence Zone; the untrained copy prefers Cartesian at every
point from 1 (−0.009 to −0.020, all intervals below zero), and at the patch embedding (point 0) no candidate explains
anything (R² ≈ 0). The competition does not show the second harmonic is needed: the pre-registered winner is polar 2-D + v (k = 1) at
24 of 25 trained points, never polar 4-D + v; on the transfer set at point 12 adding (cos 2θ, sin 2θ) to polar 2-D buys
+0.073 R² (0.334 → 0.407), less than going from two to four random smooth features of the same labels (+0.148, 0.135 →
0.283), and the best of 20 random rank-4 sets (0.473) beats polar 4-D; polar 4-D + v reaches 0.502 against 0.310 random
and 0.652 for the rank-5 knot-PC ceiling. The k ≤ 2 band limit rests on the centroid DFT (§3.2). So polar beats
Cartesian at matched rank after the zone, as one harmonic plus speed; it is not linear in a frame the model owns. The
knot-PC ceiling is the probe variance inside the knot rows' top-k PCs, not a strict bound on a rank-k map (the untrained
copy's Cartesian frame exceeds it on the transfer set at points 19–25). In steering over the 16 arcs of
the unified bake-off (§4.3), a straight edit in (cos θ, sin θ, cos 2θ, sin 2θ) has the raw chord's minimum path radius
(0.61 / 0.63 at points 12 / 22), so it still crosses the hollow; at the chord's norm its endpoint is −0.34° [−0.55, −0.13]
against the chord at point 12 and +2.13° [1.92, 2.37] at point 22 (this file's intervals; the bake-off file gives [−0.51,
−0.18] and [1.94, 2.32]); at own norm polar 2-D steers slightly better than polar 4-D (−0.13° [−0.17, −0.09] at point 12); against the knot-CV Reinsch smoother it reads 4.75°
against 6.27° at point 12 and 3.68° against 3.76° at point 22 at own norm (the earlier +0.13° tie was a 267-clip,
one-arc number). Cartesian edits fail (59° at both points) because 48% of the bake-off's carriers are accelerating clips
that start at rest, where v (cos θ, sin θ) is zero; on the constant-velocity carriers alone (post hoc, 154 clips) the
Cartesian edit beats polar 2-D at the chord's norm at both points (8.11° against 8.52°, −0.41° [−0.78, −0.03], at point
12; 8.74° against 10.31°, −1.58° [−2.11, −1.08], at point 22), and polar 4-D beats both (6.10° / 6.14°); so "polar
beats Cartesian" is an encoding result, not a steering one. Every "+ v" arm equals its arm without v in steering (v
cancels at fixed speed). The probe endpoint is partly circular for these edits, since the probe reads cos θ and sin θ linearly, so
nearest-real R is the independent test. On the speed set an edit in S_dir ⊕ S_speed hits direction and speed together
(4.2–5.1° and 0.08–0.11 m/s at points 12 / 19 / 22) where the full-space cell chord gets 13.9–14.8° and 0.20–0.21 m/s,
and adding the second-harmonic plane changes nothing. Cartesian decodes direction best at point 1 (6.95° against
12.3°), so where decoding is easiest is not where a coordinate is explicit; pixels plus frame differences decode
Cartesian direction at 15.3° but do not transfer (48.7–49.2°); start position alone is at chance. So training turns a
pixel-like Cartesian velocity code into one in which polar beats Cartesian at matched rank after the zone, as one
harmonic plus speed (the ring's k ≤ 2 band limit is from the DFT); the untrained copy stays Cartesian.

*The four-number frame through the predictor*[^f4p]. The straight edit in (cos θ, sin θ, cos 2θ, sin 2θ), run on the
session-2 carriers (200 × 4 targets) and read by the predictor: frame-wide at block 22 at the natural norm it reaches
28.6° [26.4, 30.9], against 25.7° for the chord, 12.3° for the paper's spline and 145.0° for the far-end null (unedited
92.1°, twin ceiling 9.3°), 2.9° [1.0, 4.5] behind the chord (own norm 31.3° against 27.2° and 11.3°); on the disk
tokens at block 12 it is the best arm, 33.0° [28.9, 37.6] against 40.8° for the chord, 100.2° for the spline and 116.4°
for the null (−7.8° [−10.8, −5.1] against the chord). Forced choice cannot rank arms here (it is recovery > 0.5, and
pooled edits recover only 0.15–0.26 of the twin's forecast change), so the probe-free readouts reported are recovery and
4-way twin identification: the Fourier edit identifies the right twin 0.066 [0.049, 0.085] above the null at block 22.
The positive control, the twin's cached forecast scored against this run's forecasts, reaches forced choice 0.995,
recovery 0.989 and twin identification 0.924. Extrapolation (a half-ring fit, 179 carriers, targets 5.6–45° past the
fitted end): through the predictor at block 22 (natural norm) the Fourier edit reads 32.2° against 22.2° for the spline
continued along its end tangent and 27.2° for the chord to the nearest knot (Fourier − spline +10.0° [7.0, 13.0]); at
block 12 on disk tokens 62.3° against 92.1° and 57.9°. At the encoder level, with no predictor, the half-ring Fourier edit
wins at both blocks (block 22, own norm: 5.3° against 7.6° for the spline and 25.4° for the chord; block 12: 10.7°
against 15.0° and 25.9°). Which rule wins depends on readout and site: through the predictor at block 22 the spline
continued along its tangent wins by the probe (12.2° at own norm) with an edit 1.9–3.3× the others' own norms (median
36.1–36.7 against 18.1–19.2 for the Fourier edit and 11.0–14.1 for the chord), and is worst without the probe (recovery
0.09 at own norm and 0.07 at the natural norm, against 0.17–0.18 for the Fourier edit). So the four-number frame is slightly worse than the chord frame-wide at block 22 (paired +2.9° [1.0, 4.5] at natural size, +4.0° [2.3, 5.6] at own size), is the best arm on block-12 disk
tokens at the natural norm (at own norm every disk-token arm sits at 89–103°, the unedited level), trails the paper's spline by about 16° frame-wide at block 22, and extrapolates in the encoder but not through
the predictor. No FITPACK arm was run.

**Velocity sheet v2: steering to held-out (direction, speed) cells**[^vs2]. The v1 sheet (§5) was scored at each arm's
own norm against a ring fit on one speed band. The rerun uses three hold-out designs: block2 (v1's block of 2 direction
× 2 interior speed bins), block3 (3 × 3), and cross, which hides two direction bins at every speed and two speed bins at
every direction, so the target direction and the target speed are both absent from every fitting step. Four block
positions are pooled, readers are fit on probe folds that never see a held clip, and every arm is scored at its own norm
and rescaled per clip to the sheet's norm (matched). The sheet is a thin-plate spline over (cos θ, sin θ, scaled speed)
fit to knot cell centroids, with light smoothing chosen on other blocks; the interpolating version is within 0.3–0.4°
at own norm. The fair 1-D comparator is the sequential edit (a ring pooled over speeds plus a speed line pooled over
directions; the two orders are the same additive edit) built from the paper's interpolating splines, since our
FITPACK smoother is known to lose (§4.3). Steering both variables at point 22, own norm, the sheet lands at 4.3–5.0°
across the three designs, level with the reader's own error on real target-cell clips (4.7–5.8°), and the interpolating
sequential edit at 7.0–10.5°: gaps −2.64° [−3.18, −2.14] (block2), −5.48° [−5.99, −4.98] (block3) and −6.19° [−6.73,
−5.62] (cross). The MLP reader agrees (−3.3° to −6.2°), and so does the acceleration × direction sheet at point 21
(−3.4° to −6.3° and −0.18 to −0.40 m/s²). At point 12 the block2 gap is −2.21° [−2.83, −1.63] matched. The
difference-of-means arms lose even with the true held-out centroid (8.53° against 4.33°, matched), but they start at
the source cell's centroid while the sheet starts at the clip's own (θ, v), so that gap is partly a source-point
artefact (ledger #302); probe-axis min-norm edits leave the data (nearest-real R 0.05 against 0.36–0.40 for the sheet).
The direction-only advantage of v1 (2.9° against 6.9° at point 22) was mostly our smoother's and the one-band fit's
loss: against the interpolating or Reinsch-CV one-band ring it shrinks to −1.24° [−1.63, −0.87] and −0.57° [−0.91,
−0.23], and at point 12 the Reinsch-CV ring beats the sheet (+0.33° [0.03, 0.62]). Ablations say about two thirds of the
joint advantage is statistical: a speed-independent cylinder built from the sheet already reaches 5.54° against 7.94°
for the sequential FITPACK edit and 4.33° for the sheet (point 22 block2, matched), because one smooth fit to both labels
is denoised and not confounded by each bin's speed mix. The remaining third comes from the ring's radius growing with
speed (freezing it costs 0.75° [0.47, 1.02] and 0.07 m/s); freezing the ring's shape costs nothing (4.38°). The sheet's
low speed leakage for direction edits holds at every dose from 0.5× to 2× (0.017–0.068 m/s against 0.040–0.161 for the
one-band ring), but the speed-averaged sheet matches it exactly, so that too is statistical; the dose curves were run
only against the FITPACK rings. Beyond the training maximum of 4.0 m/s the linear speed line extrapolates better at
point 22 (0.15–0.20 against 0.24–0.40 m/s) and the sheet at point 12 (0.24–0.38 against 0.32–0.63), with readers fit
only up to that maximum. Through the predictor[^vsp] (point 22, v1's block, 200 carriers, 804 GPU-s; a 48-carrier pilot
gave the same signs) the joint edit's direction advantage over composed 1-D edits survives: forecast direction error
29.8° [27.5, 32.0] against 36.2° for the sequential FITPACK edit, 34.5° with interpolating curves and 35.9° for the raw
chord (paired −6.4° [−7.6, −5.3], −4.8° [−6.2, −3.3] and −6.1° [−7.6, −4.6]; at the sheet's norm −3.9° [−5.1, −2.6],
−4.2° [−5.6, −2.8] and −4.3° [−5.7, −2.8]), with the twin forecast at 7.3° and the unedited at 103.3°. Against a
direction-only ring edit the two readouts disagree: on the pooled direction probe the ring is as good or better
(interpolating ring 28.2°; sheet minus ring +1.6° [0.1, 3.3] at own norm, +6.7° [5.0, 8.5] at the sheet's norm; +2.8°
[1.3, 4.5] and +7.9° [6.1, 9.7] against the FITPACK ring), while on the heading of the forecast's per-step disk positions
the sheet is better (−7.1° [−9.3, −5.0] at own norm, −7.5° [−9.3, −5.5] at the sheet's norm; −5.5° [−7.4, −3.6] against
the FITPACK ring); the ring edit leaves the forecast's speed 0.89 [0.83, 0.95] m/s further off than the sheet (0.86 at
the sheet's norm), which can bias a pooled direction probe. So against composed sequential edits the sheet wins on
direction, and against direction alone the answer depends on the readout. The forecast's pooled speed readout follows every arm that carries a
speed edit alike (R 0.89–0.94), and the per-step disk positions do not show the disk moving at the edited speed (pure
speed edit, R 0.03 [−0.03, 0.08]). So the sheet is a better encoder-level target for a joint edit than composed 1-D curves, mostly
for statistical reasons, the gain holds when both target values are unseen, and it does not yet produce a faster
predicted disk.

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
Where the point-12 edit goes (headline arc, 16 carriers)[^rep]: over blocks 13–16 the attention sublayers remove 0.45 and
the MLPs 0.40 of the spline edit along its own direction (16 carriers, one arc), so 15% survives block 16 and 9% block 24, against 68% of a
point-22 edit surviving blocks 23–24. A random edit of the same norm decays faster (6% after block 16, 2% at block 24),
so both edits are mostly erased, the random one more so, and the repair is not specific to the direction code: what the network removes is a frame-wide pooled perturbation, while
the direction the forecast carries still originates in the object's tokens at point 12 (§4.6, disk-token share 0.88, routed through the remaining encoder blocks).

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

At point 12 the same fit set and carriers give a negative of a different kind: no edit moves the forecast far from the
unedited 92.1°, the norm-matched chord reaches only 84.8° [77.7, 91.5], and the twin-difference edits land at
84.5–87.5° at the chord's norm and 79.5–84.1° at the natural norm. Paired against the chord at the same norm, ranks 1–2
tie it (+0.1° [−1.0, 1.3], −0.3° [−1.4, 0.6]) and higher ranks are worse (full +2.7° [1.8, 3.5]), so here error rises
with rank, the reverse of point 22. The spline and probe-QR could not be compared at matched norm, because that forward
ran without comparison-arm forecasts and the norm-matched cache covers point 22 only; the session-2 interpolating spline
at its own norm, 12.2× the chord's, gives 77.9° [73.8, 82.3] (label-free knot order; 81.8° on the labels order). At the encoder output the twin edit ties probe-QR at the
chord's norm (+0.2° [−0.3, 0.7]) and is 1.8° worse at the natural norm, and all arms stay above 80°. The point-12 twin
comparison is therefore twin versus chord only, on one fit set[^twd].

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
dose; for direction it is not (next paragraph). Through the encoder (full clip, own norm, a speed probe refit at every later point) point-12
edits keep R 0.90 at point 12, 0.44 two blocks later and 0.13 at point 25. Point-22 edits keep 0.57 at point 25 (0.42 m/s
from target, against 1.27 unedited and 0.10 for the twin). So speed behaves as direction does in where edits survive
at their own norms, and the spline adds nothing over the chord, as the straight geometry predicts (§4.1). The forecast's
speed code moves, but the forecast disk does not move faster, so this is a probe-level change in the forecast and not a
faster predicted disk (`figures/fig_session3_speed.png`).

*Direction at point 12 at the natural dose*[^natdir]. The session-2 point-12 direction edits, rerun with each edit
rescaled to the natural twin change (median norm 14.2; the natural norm is the twin's change in the full-clip meanpool, while the edit is applied to the context-only encoding) on the same 200 carriers × 4 targets and the same predictor-side
probe (the own-norm rerun reproduces session 2 exactly), leave the forecast near the unedited 92.1°: the chord reaches
79.7° [72.5, 86.6] (R 0.29), the label-free interpolating spline 87.9° [80.9, 94.5] (R 0.05) and the null, the same edit
aimed at the far end (143.4°), 102.3°, against 9.1° (R 0.96) for the rendered twin and 12–26° for the same edits at
point 22. The natural dose moves the point-12 forecast by only −5.1° (chord) and +10.0° (spline) relative to own norm,
unlike speed (R 0.57). The encoder removes the edit in proportion: 17.5% of the chord edit survives to block 16 and 11%
to block 24 at both doses, against 69% at block 24 for a point-22 edit. The one point-12 edit that does reach the
forecast puts the chord direction on the disk tokens alone (9% of context tokens), each at the twin's own per-token
change: 40.8° [35.6, 46.3] (R 0.62), with the null at 116.4° and the spline, whose direction differs, at 100.2°. It
still falls well short of the twin. Location matters as well as dose[^bgctrl]: the same edit on as many
background tokens (median 87) with the disk tokens' own per-token magnitudes, permuted, leaves the forecast at 88.9°
[81.7, 95.6] (R 0.09; 48.0° [41.9, 54.6] worse than the disk tokens, paired), and the same total energy spread over all
background tokens (median 937) reaches 80.1° [73.2, 87.1] (R 0.25; 39.3° [32.9, 46.0] worse), no better than the pooled
frame-wide chord at the natural norm (79.7°): it moves the forecast about 12° from the unedited 92.1°, as much as a frame-wide edit does, and no more. The disk-token edit is also specific to point 12: the same construction at point 16
reaches 67.0° [61.6, 72.6] (R 0.31) and at point 19 70.6° [63.6, 77.6] (R 0.59), 26.1° [20.9, 31.3] and 29.8° [22.5,
36.9] worse than at point 12 despite a larger edit energy (median 4.3 × 10⁵ and 6.3 × 10⁵ against 3.2 × 10⁵). So at
point 12 a direction edit reaches the forecast when it is put on the disk's tokens and not when the same energy is put
anywhere else; that is a location effect on 200 carriers, and it still recovers under two-thirds of the twin's move
(40.8° against 92.1° unedited and 9.1° for the twin). The token set is not an oracle, and the per-token dose
carries only a little of the twin's geometry beyond the disk's footprint (below)[^dsweep]. The disk-token set was the union of the source clip's disk and the target
twin's disk (plus a one-patch ring), so it carried where the disk will be under the target motion. On the source disk
alone (median 76 tokens against 87) the same edit reaches 44.7° [38.8, 50.4] (R 0.60), 3.8° [2.3, 5.3] worse than the
union, and the tokens the twin's location adds do nothing on their own (91.5°, R 0.02), though they are few (median 12
tokens, 17% of pairs with none, 0.115× the source set's edit energy) and add 3.8° jointly. What the twin still supplies is
the per-token dose pattern, since every source token gets the target twin's own per-token change: one uniform dose on
the same source tokens (0.83× the energy) reaches 65.8° (21.1° [17.1, 25.6] worse). A dose profile that uses no twin
information[^tfree], the mean per-token change of the probe-split clips at the target angle (7–8 clips per angle;
carriers are never in the probe split) rescaled to the same mean per-token magnitude (its edit energy is 1.13× the uniform arm's and 0.93× the per-token arm's), reaches 49.6° [43.5, 55.5] (49.5° with the mean
over all 480 probe clips, which carries no target information; the two differ by 0.06° [−0.08, 0.23]). It recovers 77%
[70%, 82%] of the uniform-to-per-token gap and leaves 4.9° [3.7, 6.3] to the twin's own profile; unrescaled (a larger dose,
per-token mean 73.6 against 56.7) it matches the per-token arm (45.0°, +0.3° [−2.2, 3.0]). So most of the per-token
advantage is the spatial shape of the dose, the disk's footprint, not the twin's geometry; a small residual is
twin-specific. Dose
matters in both directions: R rises 0.30 / 0.60 / 0.82 at 0.5× / 1× / 2× the twin's per-token change, but the error is
76.2° / 44.7° / 49.5° (at 2× the error rises by 4.8° [0.2, 9.4] while R is still 0.82 [0.73, 0.90] < 1, so the extra dose
goes off-axis rather than past the target). On the source tokens the straight edit in (cos θ, sin θ, cos 2θ,
sin 2θ) is the best arm, 37.0° [32.1, 42.3] (R 0.71; 7.7° [5.1, 10.6] better than the chord). The paper's interpolating
spline fails there on the label-free knot order (100.4°, worse than unedited; own norm 88.5 against 9.2 for the chord)
and recovers on label-ordered knots (68.1°, −32.2° [−37.0, −27.5]; own norm 14.1), still 23.5° [19.4, 27.7] behind the
chord (66.3° on the union set). By depth, source-disk edits reach 86.0° at block 8 (inside the Physics Emergence Zone),
44.7° at block 12, 69.1° at block 16 and 76.5° at block 19, so block 12 is where object-token edits work best.

*Acceleration through the predictor (GPU session 3b)*[^s3a]. The session-3 design on the acceleration set: 128 test
carriers, each steered to 4 targets in the held-out 7.52–8.61 m/s² block with the smoothing spline, the chord between
smoothed knots, the raw-centroid chord and a null aimed at 0.25 m/s², at points 21 and 12, at own, chord and natural
norms, read by probes fit only on unedited forecasts of probe clips outside the block. A direct acceleration probe reads
the forecasts well (test R² 0.92); a per-step position readout does not (−0.18, against 0.91 from the true centroids),
so only the direct probe is reported. At point 21 the own-norm spline edit cuts the error to target from 3.46 to 1.82
m/s² (R 0.39); at the twin's norm it reaches 1.32 m/s² but overshoots (R 1.82), against 0.83 m/s² for the twin's own
forecast. Spline and chord are indistinguishable (paired −0.010 [−0.012, −0.007] m/s² at own norm, +0.012 [0.010,
0.013] at the chord's), as the straight geometry predicts; the null moves the forecast away (6.97 m/s²). Point-12 edits
mostly wash out at own norm (R 0.14) but reach R 0.88 at the natural norm (2.38 m/s²), so for acceleration, as for
speed, the point-12 loss is partly dose. The point-21 edit raises the forecast's mean speed by 0.25 m/s where the physics
of the target predicts 1.41 m/s (the twin's forecast: 1.46), so the predictor reads the edit as a partly decoupled
acceleration label rather than a physical change of trajectory.

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

*Without the heading probe*[^f4p]. The Fourier-4 run scores the block-22 forecasts three probe-free ways against the
rendered twin's forecast (positive control: the twin's own cached forecast reaches forced choice 0.995, recovery 0.989,
twin identification 0.924). Every arm stays nearer the source's forecast: at the natural norm forced choice is 0.059 /
0.094 / 0.116 / 0.016 (spline / chord / Fourier / null), whole-forecast recovery 0.186 / 0.234 / 0.261 / 0.112 and 4-way
twin identification 0.331 / 0.312 / 0.315 / 0.249 (own norm: recovery 0.172 / 0.154 / 0.161 / 0.084, twin identification
0.331 / 0.310 / 0.315 / 0.250). So the edits move the whole forecast a fifth to a quarter of the way to the twin's and
point it at the right twin only slightly above chance, and the probe's ranking (spline 12.3° against chord 25.7°) is only
partly reproduced probe-free: mixed, with the spline ahead on all three readouts at own norm and the chord ahead on
recovery at the natural norm. Every forecast-level claim in this section is about the forecast's
heading code as read by the predictor-native probe. Recovery and forced choice also reward a large, non-specific edit:
at own size the block-12 disk-token spline edit (norm 88.5 against 9.2 for the chord) has the highest forced choice of
any arm at any site, 0.245 [0.204, 0.284], and a recovery of 0.255 (second only to the natural-size Fourier edit at block
22, 0.261), while its heading is 103° from the target and its twin identification, 0.242, is at chance. So the own-size
"spline leads probe-free" part of the mixed verdict is confounded by its larger edit (about 1.4× the chord's own norm at
block 22; ratio of medians 1.37, per-pair median 1.39, mean 1.54). At matched size[^pfms] the chord wins forced choice and
recovery and the spline keeps twin identification: at block 22 on all tokens, with both edits at the twin's norm
(median 19.07), the chord beats the spline on forced choice (0.094 against 0.059; paired −0.035 [−0.052, −0.019]) and
recovery (0.234 against 0.186; −0.048 [−0.057, −0.040]), while the spline keeps twin identification (0.331 against
0.312; +0.019 [+0.001, +0.037]) and the heading-probe lead (12.3° against 25.7° at that size); in the separate own-size ladder
contrast, with the spline at its own size and the chord interpolated to it, twin identification is +0.036 on the 176
carriers (about 523 carrier–target pairs) whose sizes fall inside the chord rungs. On block-12 disk tokens at matched
per-token dose (identical per-token magnitudes for both arms) the spline leads on forced choice (+0.087 [+0.043, +0.126])
while its heading is 100.2° off against the chord's 40.8°, and it loses on twin identification (−0.024) and on heading
by 59°: forced choice rewards a heading-wrong edit even at equal size, so forced choice is not heading-specific. The
own-size ladder at block 12 (a different token set and dose shape for the chord rungs) gives +0.034 [−0.016, 0.081] on all
pairs and +0.040 [−0.017, 0.095] in bracket. Caveats: the ladder interpolates linearly between two chord sizes at block 22
and three at block 12 and extrapolates 34.6% of block-22 pairs and 49.75% of block-12 pairs beyond the run sizes (the
quoted ladder values are all-pairs; the in-bracket subsets agree in sign), and the reference twin forecasts are the
Fourier-4 rerun's (the cache discrepancy in §7 is unresolved).

*Writing a contact into the encoder*[^csteer]. The 96 straight twins of the in-context contact set (§5) are edited on
their post-contact context slots with a bounce direction: the mean bounce-minus-straight difference of the other clips
with the same wall and approach side (leave one pair out), read by a transfer reader that never saw an in-context clip.
At block 22 the edit on all tokens turns the forecast 0.48 [0.40, 0.55] of the way to the reflected heading (unedited
−0.00 [−0.03, 0.02]; real bounce clips 0.94 [0.87, 1.01] with this reader), and on the disk tokens only 0.03. At block 12
the pattern reverses: all tokens 0.05, disk tokens 0.45 [0.40, 0.50]. A random direction of matched size does nothing at
either place (−0.02 to 0.00), and a difference taken from clips with a far-off heading turns the forecast the wrong way
(−0.27 [−0.31, −0.22] on all tokens at block 22; −0.20 [−0.24, −0.17] on disk tokens at block 12). The carrier's own twin
difference gives the same 0.48 at both places as the borrowed one; pasting in the twin's own bounce tokens (an upper bound)
reaches 0.82 / 0.90 (blocks 22 / 12). Early contact turns the forecast more than late (0.53 against 0.39, block 22), and
the encoder's own context reading moves with it (0.62 for the block-22 all-token edit against 0.67 for real bounces). The deciding control is a heading change on the same slots with no wall (straight-out minus straight-in, matched in
norm)[^chead]: at block 22 on all tokens it turns the forecast 0.77 [0.68, 0.86], more than the bounce edit's 0.48
(paired −0.29 [−0.38, −0.21]), and at block 12 on disk tokens 0.46 against 0.45 (paired −0.01 [−0.03, 0.01], a tie). So
the edit writes the post-contact heading, not a contact: the event itself is not shown to be steerable, and the
block-12 disk-token route carries heading only. The depth-and-token pattern (object tokens at block 12, frame-wide at
block 22) is the direction finding again (§4.6). The reading results stand: an observed reflection is carried forward
(0.84) and a future one is not anticipated (0.07; §5).

### 4.6 Which tokens carry direction, by depth (token-source patching at the predictor)

The paper's own future-work hypothesis (l.795–798) is that velocity is "bound" to the object in the middle of the
network and that the end of the network is "catered to the optimization objective of predicting the next frame in
latent space". Token-source patching tests a related question: which tokens at depth k are the causal source of the direction the forecast ends up carrying. It does not run the paper's own test (C.1.4, l.776–777: a predictor trained from scratch on each layer's features), which was not run here. For 64 random pairs and 64 position-matched pairs of
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

The encoder-output reader gives the same curve within 0.03 at every point. A patch at point k passes through encoder
blocks k+1…24 before the frozen predictor sees anything, so the fraction measures routing through the remaining
encoder, not a predictor-specific read. Up to point 12 the direction the forecast carries originates in the disk's own
tokens: swapping 9–13% of the tokens moves the forecast 88–99% of the way,
swapping the other 87–91% moves it 1–12%. Between points 12 and 22 that reverses. At point 22 the disk tokens carry
0.18–0.22 and the background 0.77–0.84, about twice the disk's token share (0.09–0.13) and near the background's (0.87–0.91), so by the
last blocks the direction code is spread across the whole frame rather than held by the object. The position-matched
pairs rule out the disk's location as the carrier. Read against the paper's C.1.4 hypothesis (velocity "most bound" to the object in the middle layers), the disk-token share falls monotonically from the input, 0.99 / 0.98 / 0.88 / 0.76 / 0.22 at points 0 / 8 / 12 / 16 / 22: there is no mid-network binding peak; the code is object-bound from the start and delocalises late, which agrees with the two-disk cross-disk index in §5 (lowest over points 6–17) rather than with the hypothesis as stated. An untrained control follows. The patching result also says why edits at point 22
reach the forecast when a per-token object code would not: a pooled edit at 22 lands on the tokens that carry the
direction into the encoder output the predictor receives. Provenance: forward on box 1 at commit 8a54162, clean tree; scored locally; 480 probe clips per reader,
5-fold ridge, CV R² 0.91 (forecast) and 0.94 (encoder output). Limits: one object, one render, five points, pair
bootstrap CIs over 64 pairs; a swapped disk also swaps the disk's appearance, which is identical across clips here.

*Untrained control*[^tokr]. The same plan (pairs and token sets) on the untrained V-JEPA 2 copy, with both readers
refit on its own outputs, gives no object binding: on the forecast reader the disk tokens carry 0.33 [0.18, 0.49] /
0.40 [0.28, 0.51] / 0.27 [0.01, 0.46] of the move at points 8 / 12 / 22 and the background 0.57 / 0.60 / 0.61, and on
the encoder-output reader the disk tokens carry 0.58 / 0.54 / 0.43 and the background 0.44 / 0.47 / 0.58 (trained:
disk 0.98 / 0.88 / 0.22, background 0.04 / 0.12 / 0.77 on the forecast reader); position-matched pairs look the same.
The untrained readers are weak (CV R² 0.35 for the forecast and 0.59 for the encoder output, against 0.91 and 0.94),
so these fractions are noisier. So the object binding through point 12, and most of the late handover to the
background, are made by training; the architecture alone gives at most a small drift toward the background (disk
0.58 → 0.43 on the encoder-output reader).

**The same question from the encoder side (pooled object, background and scene tokens)**[^ovs]. Before the
predictor, does the encoder itself keep direction on the disk? Three pools per clip, built exactly from the stored
time-pool, disk-pool and disk-mask arrays: the mean of the disk tokens (object), the mean of every other token
(background) and the stored mean over all tokens (scene). Accuracy cannot tell them apart: at point 8 a ridge probe
reads direction at R² 0.987 from the object pool, 0.978 from the background and 0.979 from the scene (MAE 3.1°, 4.7°
and 4.5°), and at point 22 0.995, 0.991 and 0.992; the untrained copy's scene pool stays at 0.87–0.88 at every point.
What separates them is the axes. A probe fit on the object pool and read on the background pool transfers with R² 0.27
at point 8 and 0.17 at point 12 but 0.77 at point 22 (background→object 0.72, 0.44 and 0.88), and the angle between
the object and background ring planes is 78–89° at points 8 and 12 and 53–56° at point 22. So through point 12 (after the paper's zone at blocks 8–9) the disk tokens and the rest of the frame both decode direction but on nearly orthogonal directions of the
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

### 4.7 Do heads that attend to the disk's previous position carry the direction code?

This is our disk-attention test, not the paper's §6.3 / App. C.6 attention-distance metric, which was not run (it needs
a rerun of the attention pass, and C.6 does not specify how per-head distance is aggregated, so that metric is itself
under-specified)[^heads]. Nor was the paper's causal test run: suppressing local attention in the emergence zone, by
masking weights to nearby tokens and renormalising the remainder (C.6 l.1026–1027), and reading direction R² (Tab. 2:
0.97 → 0.83 with temporal masking t = 3, 0.14 with spatial s = 3 plus temporal). Our head-ablation null neither tests nor
contradicts that result. *Part A* scores every head at blocks 6–13 on 300 train clips: queries are the
disk's tokens at slot t, and the score is the attention density on the disk's tokens one slot earlier over the density on
background tokens, with same-slot, next-slot and far-slot (≥ 2 slots away) ratios beside it and a phantom control (the
disk mask rolled by 8 × 8 patches, the same trajectory shape with no object there). By default a head is labelled
"object-attending", since a head that attends to the object wherever it is also scores high; it is labelled "tracking"
only when its previous-slot over same-slot ratio has a clip-bootstrap interval entirely above 1. In V-JEPA 2, 127 of 128
heads put more than twice, and 125 more than five times, the background density on the previous-slot disk tokens (block
medians from 37.5× at block 13 to 75.0× at block 8; the top 6 heads hold 32% of the excess), against 1.07–8.95× for the
phantom mask. The "tracking" label uses the previous-over-same ratio only, and it is weak: 19 of 128 heads pass it, and so do
69 heads of the untrained copy at prev/same ratios of 1.00–1.15, artefacts of two near-background densities with tight
intervals (the untrained copy has no attention structure: block medians 0.90–1.04×, maximum 1.51×, no head above 2×), so
trained and untrained head counts are not compared. Requiring the previous slot to beat the same, next and far slots and
the phantom mask leaves 9 heads; only three of them put more than ten times the same-slot density on the previous slot,
block 7 head 15 (previous-slot 1,052× [1,000, 1,108], previous over same 13.3 [12.9, 13.8]), block 11 head 6 (989×, 13.3)
and block 13 head 8 (116×, 13.3), and their phantom-to-real ratios are 0.58 / 0.29 / 0.12, so only block 13 head 8 is
mostly object-specific. We quote 19 under the weak criterion and 9 under the strict one, of which 3 have prev/same > 10.

*Part B* zero-ablates heads (their slice of the input to the attention output projection) at blocks 8, 9, 10 and 12 and
reads the fixed Part 1 probe at point 12 and point 22 on the 300 direction test clips (and 154 of the 308 speed test clips). The
unablated baselines, 0.9882 (point 12) and 0.9918 (point 22), match the Part 1 fp32 scores (0.9885 / 0.9919). The eight
heads with the highest previous-slot ratio across those blocks (block 9 head 9, 12/6, 10/4, 8/2, 10/12, 12/13, 10/6 and
8/15; only 10/4 is labelled tracking) cost 0.040 [0.035, 0.045] of the point-12 direction R² when ablated together, less
than each of four count-matched random head sets (0.124 / 0.174 / 0.207 / 0.220), and the per-block top three sit inside
or below the random range at every block (block 10: 0.009 against 0.012–0.145; block 12: 0.095 against 0.065–0.376). The largest single-head drop is block 12 head 4 (0.312 [0.297, 0.331]), a head
the attention score did not select. At point 22 every single head costs at most 0.0054 and the eight heads 0.0051
[0.0037, 0.0066] (random sets 0.0019–0.0066). Speed behaves the same way (largest single head block 12 head 2, 0.094;
the eight heads 0.059 [0.048, 0.073] against 0.009–0.582 for random sets). Zero-ablation read by a probe fit on unablated
activations mixes information loss with distribution shift, and the random sets are only a partial control for that.
The untrained copy's Part B is invalid: in bf16 its unablated baseline does not reproduce the stored fp32 probe score
(direction R² −15.7 against 0.876, because the random-init probe amplifies the ~0.5% feature error), the file flags it,
and an fp32 rerun was not run, so there is no untrained ablation control. The result is a null: the heads with the
highest previous-slot density at the ablated blocks matter no more than random head sets of the same size, and the
probed direction code is not concentrated in a few heads. Four of the nine strict-criterion heads sit in ablated blocks
and were zeroed alone (block 8 head 14, 9/4, 10/4, 12/3), costing 0.011 / 0.008 / 0.017 / 0.016 of point-12 R², within
each block's single-head range; only the three with prev/same > 10 (blocks 7, 11 and 13) were never ablated, and ablating
them is a named next step. Note too that the score is previous-slot density over background: 108 of 128 heads put more
density on the disk's current slot than on its previous one (block 9 head 9: 4,065× current against 683× previous), so
these are heads with high attention to the disk, not heads that prefer the previous slot.

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
| Velocity sheet (direction × speed, steering) | The sheet is neither cone nor cylinder: ring radius grows with speed to ~1.4–2.4 m/s then flattens (point 12 noise-corrected radius 3.17 at 0.46 m/s, 7.70 at 3.79 m/s; fast ÷ slow 2.43 [2.06, 2.57] against a speed ratio of 8.29; linear-fit intercept 4.38, a cone needs 0); held-out-cell model fit: cylinder 0.804, radius-scaled ring 0.834, sheet 0.843, noise ceiling 0.910. Steering to held-out cells, each arm at its own norm (v1 had no norm matching: edit norms 10.21 / 9.60 / 9.78 for sheet, ring and chord at point 12), clip-bootstrap CIs: direction at fixed speed, sheet − one ring per speed band (fit on one band's knots, about 1/8 of the data) −2.47° / −2.58° / −4.02° at points 12 / 19 / 22, sheet − raw chord −0.42° / −1.02° / −1.50°; both variables, sheet − two sequential 1-D edits (the two orders are the same additive edit) −1.66° / −2.16° / −4.35° and −0.060 / −0.049 / −0.031 m/s; off-target direction change of a speed edit, sheet − 1-D speed line −1.3° to −1.8°. The untrained copy has no ring (radius ≈ 0.03). v2 (§4.4), matched to the sheet's norm, with pooled baselines and a cross design where target direction and speed are both unseen: direction to the held-out cell, sheet − sequential global ring + speed line −3.60° [−4.12, −3.07] (point 22 block2), −4.05° (cross), −1.50° (point 12); direction only, sheet − global ring pooled over speeds −2.72° [−3.34, −2.09] with our FITPACK smoother but −1.55° [−2.00, −1.14] with a Reinsch-CV ring (point 22), so most of v1's 4.0° ring gap was a data-starved baseline; cylinder costs 1.2°; the forecast keeps 3–6° of the joint advantage and shows no speed advantage. | `p5_velocity_sheet.json` (`shape`, `model_fit`, `steer.*.sheet_minus_*`), `p5_velocity_sheet_v2.json`, `p5_velocity_sheet_predictor.json`[^vs2], `fig_velocity_sheet.png`, `fig_velocity_sheet_v2.png` |
| Speed and acceleration beyond straight lines | The acceleration line is mostly the mean-speed line: at point 12 the two lines have cosine 0.92 (0.90–0.95 at points 8–25, 0.99 at point 4; split-half reliability 0.99), and a speed probe reads accelerating clips at 0.295 s × a, close to their physical mean speed of 0.3125 s × a; the untrained copy transfers the same way (slope 0.28, line cosine 0.57–0.65), so this part comes from the stimulus. Training adds an acceleration code that separates accelerating from constant-velocity clips at matched mean speed (AUROC 0.996 at point 12, ≥ 0.999 from point 16; untrained 0.53). In the per-step tokens speed is mostly a rate code: 79–93% of the per-step speed vector is the same at every step, the part growing with elapsed time is 1–7%, and the speed line is orthogonal to the within-clip time direction from point 12 on (abs cos 0.009–0.036 against a random p95 of 0.06); a step-specific edit moves decoded distance 0.90–0.95 of Δv·τ, while a pooled edit, as in all our steering, shifts it by a constant 0.31 m, which may be why the Session 3 edit moved the forecast's speed probe but not the disk. Speed edits along the global line transfer across directions on the probe (gain 0.98–0.99, 0.75–0.96° of direction change) but lose nearest-real agreement away from their own bin (own minus 90°: 0.057 [0.052, 0.061] at point 12), so the axis is shared for readout more than for steering. Spacing along the line slightly prefers log speed (§4.1 corrected), and a real doubling of speed inside the range puts only 0.17–0.20 of its activation change on the line though the probe reads 0.97–0.98 of it. Encoder-side, pooled results; the twins come from a different stimulus family (paper layout). Through the predictor, a point-21 acceleration edit moves the forecast's acceleration readout (3.46 → 1.82 m/s², R 0.39; spline = chord) but raises its mean speed only 0.25 of the 1.41 m/s physics predicts (§4.5). Acceleration decorrelated from mean speed[^adec] (the target here is signed acceleration, decelerations negative; the paper reports a Cartesian (ax, ay) target (§5.2) and the magnitude |a| (§5.3), both reported after this row's signed numbers; 240 rendered clips, 4 mean speeds 1–3.25 m/s × 5 accelerations −3 to +3 m/s² × 12, correlation with mean speed and displacement 0.000; 5-fold CV, clips of each cell spread over folds): with mean speed and displacement partialled out, V-JEPA 2's mean-pooled vector reads acceleration at R² −0.01 / −0.03 / 0.37 / 0.27 / 0.44 [0.33, 0.53] / 0.75 / 0.79 at points 1 / 4 / 8 / 9 / 12 / 16 / 22 (per-tubelet 0.53 / 0.50 / 0.65 / 0.81 / 0.85 / 0.87 / 0.89) and the untrained copy at about −0.01 everywhere, so training adds acceleration beyond mean speed and displacement, and in averaged features it appears at point 8, at the zone. The audit's cell-level bootstrap, which resamples the 20 design cells rather than treating the 12 clips of a cell as independent, gives wider intervals[^aaudit]: mean-pooled 0.37 [0.10, 0.46] at point 8, 0.29 [0.00, 0.39] at 9 (marginal), 0.44 [0.17, 0.56] at 12 and 0.79 [0.67, 0.84] at 22; per-tubelet 0.53 [0.34, 0.59] at point 1. Our own cell-block bootstrap of the same partialled fits[^acell] (1,000 draws over the 20 cells) agrees: signed a is about 0 in averaged features at blocks 1 and 4 (−0.008 [−0.312, −0.005]; −0.027 [−0.351, 0.002], interval including zero), as the claim expects, and from block 8 on it excludes zero at every trained point except averaged block 9 (0.27 [−0.03, 0.37]; block 12 averaged 0.44 [0.12, 0.55], block 22 0.79 [0.67, 0.84]). Both targets are constant within a cell, so cell resamples that draw few distinct levels drag R² down, and these intervals are conservative. Held out a whole design cell at a time[^acart], signed acceleration holds (mean-pooled 0.42 / 0.53 / 0.83 at points 8 / 12 / 22; per-tubelet 0.65 at point 1); held out a whole mean-speed level, it falls to 0.25 [−0.16, 0.40] at point 12 (mean-pooled, cell bootstrap). A permutation null within mean-speed level (200 permutations) has a 95th percentile of 0.002–0.009 (maximum 0.087), and a planted linear signal is detected 60% of the time at oracle R² 0.014 and 90% at 0.066, so the ≈ 0 at points 1–4 mean-pooled is a tight null for a linear code. Design caveats: acceleration correlates −0.62 with initial speed and +0.62 with final speed, so given mean speed this design cannot distinguish acceleration from initial or final speed, and what is read is acceleration beyond mean speed and displacement (on the grid displacement is mean speed × 0.625 s exactly, so this means beyond mean speed); |a| ≤ 3 m/s² covers only the bottom ~30% of the supplied 0.25–10 m/s² range and of the paper's 2–10; a small heading-dependent position cue exists (r −0.24, at most 3 px); renderer parity IoU 0.984, disk area 346 against 350 px. Once the in-set decoded per-step speed sequence (per-step probe fit inside each fold, R² 0.84–0.96) is partialled out, nothing is left (−0.01 to 0.00 at points 4–22 mean-pooled, −0.04 to +0.02 per-tubelet), and an MLP on activations (0.31–0.53 per-tubelet at points 8–22) never beats the same MLP on the decoded speeds (0.86–0.94; paired difference −0.40 to −1.02, every interval below zero); an MLP on the true per-step speeds reaches 0.999 and on true mean speed alone −0.004. This test cannot separate a direct acceleration code from a speed-sequence code: acceleration is an exact linear function of the per-step speeds, and a direct (v0, a) code would also decode each step's speed linearly, so removing eight in-set decoded speeds zeroes acceleration by construction. With a speed decoder fit on the constant-speed set instead (out of set; renderer shift is a caveat), per-tubelet features keep 0.32 [0.17, 0.44] / 0.46 [0.33, 0.56] and the mean-pooled vector 0.07 [−0.06, 0.17] / 0.26 [0.08, 0.40] at points 12 / 22 (untrained about 0), which leaves room for a direct code. Caveat: 240 clips against 1,024 or 8,192 features leaves the activation MLP data-starved. So acceleration is readable beyond mean speed once decorrelated; whether it is read directly or through the speed sequence is undecided, and the paper's "single MLP without a velocity intermediate" claim cannot be tested as designed. On the paper's §5.3 target, the magnitude |a|[^amag] (on this grid uncorrelated with mean speed, displacement and signed acceleration; 48 clips at |a| = 0, 96 at 1.5 and 96 at 3 m/s²), partialled linear R² is about 0 through block 12 in both feature types (mean-pooled 0.00 / −0.03 / −0.01 / −0.03 / −0.01 at blocks 1 / 4 / 8 / 9 / 12; per-tubelet 0.01 / 0.02 / −0.03 / −0.05 / −0.03) and weak after (mean-pooled 0.12 [0.01, 0.21] and 0.16 [0.04, 0.26] at blocks 16 / 22; per-tubelet 0.07 and 0.13 [0.03, 0.23]); under the cell-level bootstrap the late mean-pooled values have intervals [−0.22, 0.26] and [−0.21, 0.35] (our own cell-block bootstrap: 0.12 [−0.22, 0.26] and 0.16 [−0.24, 0.36][^acell]), so |a| is weak and not distinguishable from zero anywhere on the grid; the |a|-via-signed and MLP readouts keep clip intervals only, because their out-of-fold predictions were not saved, and leaving a whole cell out gives the same (0.11 / 0.17 at blocks 16 / 22). The paper's Cartesian (ax, ay) target[^acart] is about 0 mean-pooled through block 12 and 0.17 / 0.46 at blocks 16 / 22 (per-tubelet 0.09 at block 8, not clear of zero under the cell bootstrap, and first clear at block 9, 0.165, up to 0.62 at 22; leave-one-cell-out 0.16 / 0.46 mean-pooled). Read instead as the calibrated magnitude of the per-tubelet signed prediction (calibration on held-out training clips), |a| reaches 0.06 / 0.07 / 0.22 / 0.44 / 0.52 / 0.57 / 0.62 [0.54, 0.68] at blocks 1–22. |a| minus signed a is negative with the interval below zero at every point from block 8 in both feature types (per-tubelet −0.88 [−0.92, −0.84] at block 12) and per-tubelet also at blocks 1 and 4 (−0.52, −0.48); the untrained copy is −0.02 to 0.00 in every cell. The MLP readouts are negative in every cell, untrained included (240 clips against 8,192 features), and support nothing; they are kept in the file only for protocol matching. So on decorrelated clips |a| is not readable before the zone, becomes readable through the signed code at the zone (0.22 / 0.44 at blocks 8 / 9) and only weakly by a direct linear probe late; signed acceleration is readable from block 1 in time-ordered features, and what survives of the acceleration claim is that acceleration beyond mean speed appears after the Physics Emergence Zone, direct code or speed sequence undecided. | `p5_speed_accel_angles.json`[^sacc], `session3_acceleration_predictor.json`[^s3a] |
| Within-clip time (step index) | Each of the 8 time steps has its own pooled feature; the clip-mean-subtracted residuals trace a shared curve that explains 0.82 of held-out within-clip variance at point 22 (a straight line in t 0.15; `geometry["speed/vjepa2/timepool"]["22"].heldout_var_explained` centroids 0.823, line_in_t 0.151). The coordinate advances 1.00 per step in slow, mid and fast bands alike (fast ÷ slow 1.005 [1.002, 1.009] at point 22, where a distance counter predicts 6.02), so it is a speed-invariant frame count, not an odometer. Steering to a held-out step: time-probe error 0.106 (spline) / 0.082 (raw chord) steps at point 22 against 2.43 unedited, spline − chord +0.024 [0.023, 0.025]; side effects ≤ 0.007 m/s and ≤ 0.29°. Control: the untrained copy also decodes t (test R² 0.9735 at point 22, `controls.speed.random.timepool.decode.22.test.t_r2`), so the step index itself is available to any encoder, as a positional index: the frame shuffle below shows the decoded step follows the token slot, not the frames' content, in both copies (V-JEPA 2 adds no position vector to the residual stream, so the index enters through RoPE in attention; at point 0 nothing is decoded); what is specific to training is the shared low-dimensional curve (untrained: 0.007 of within-clip variance) and the 0.99–1.00 per-step advance (on the full 1,536-clip control set the untrained copy advances 0.85–0.87 per step and is also speed-invariant, fast ÷ slow 0.99 [0.93, 1.06]; the earlier 768-clip subset gave 0.71–0.77)[^timef]. At a fixed frame rate the design cannot separate frame count from elapsed seconds, and the time-rescaled "clock" stimuli are speed changes in disguise (`CLOCK_NOTES.md`, #251/#277). Static-disk control[^tstatic] (96 zero-speed renderer twins of held-out speed clips, disk centroid constant to 6e-14 px, against their 96 moving twins): frame counter, not odometer. Static clips still advance along V-JEPA 2's time code at 1.07 [1.05, 1.08] of the moving rate on the mean time direction, 0.55 [0.52, 0.59] on the spline coordinate and 0.65 [0.64, 0.66] on the mid-band time probe at point 12, and 0.69 [0.67, 0.70], 0.83 [0.80, 0.87] and 0.52 [0.51, 0.53] at point 22 (static ÷ moving twin, ratio of means, clip bootstrap), never near an odometer's 0; the untrained copy advances at 1.02–1.10 of its moving rate (four of the six intervals include 1). The spline and probe ratios below 1 come from two steps: the static clips' per-step spline coordinate sits within 0.11 of the frame index at steps 0, 1, 3, 4 and 6 at point 12 (0–4 and 6 at point 22) but falls back to 2.6 / 2.9 at step 5 and to 3.1 / 5.8 at step 7 (points 12 / 22), where the step-centroid curve folds back and the nearest-point projection is ambiguous; so the sub-unit ratios are a projection artefact as much as a slower count. Static clips are also off the moving-clip manifold: nearest-neighbour distance 2.30× [2.23, 2.37] (point 12) and 2.90× [2.75, 3.07] (point 22) that of the moving twins, and only 0.45 / 0.56 of their within-clip variance lies on the shared curve (moving twins 0.79 / 0.80), so the static readout is extrapolation off the data (`figures/fig_time_static_control.png`). Frame shuffle[^tshuf] (308 test clips, whole tubelets permuted into three derangements with no tubelet left in place; reader fit on unshuffled clips): the decoded step follows the slot, not the content. Regressing the decoded step on slot and original content gives slot 0.956 [0.953, 0.958] and content 0.002 [0.000, 0.004] at point 12, slot 0.918 [0.915, 0.922] and content 0.002 [−0.002, 0.005] at point 22 (the decoded step is nearer the slot than the content on 99.9% / 98.8% of tubelets), and the untrained copy is just as positional (slot 0.980 / 0.981); at point 0 nothing is decoded (R² −0.003) and at point 1 the slot coefficient is 0.45 (nearer the slot on 69% of tubelets). So the step code is a token-position (frame-index) code in both copies; training changes its scale and shape (the shared curve above, and a slope below 1 on static clips), not its source. On the 96 static twins the shuffle reader's slope on slot is 0.75 / 0.62 (points 12 / 22) in V-JEPA 2 against 0.96 / 0.98 untrained. | `p5_time_manifold.json` (`geometry`, `clock`, `decode`, `steer`, `controls`), `p5_time_manifold_controls.json`, `p5_time_static_control.json`, `p5_clock_test_linearity.json`, `results/CLOCK_NOTES.md` |
| Time through the predictor | 128 held-out speed-set clips, edits on every context token. The real reference works: the same clip's context window two tubelets later advances the forecast disk 1.15 [1.05, 1.24] of the true two-step displacement (15.5 px) while leaving the context time readout unchanged (+0.01 steps), because the re-encoded window gets token positions 0–3 again. A +2-step spline edit does not: at point 22 it moves the forecast −0.21 [−0.42, −0.01] of two steps, no differently from the same path aimed backwards (+0.02 px [−1.97, 1.84]), while pushing the disk 14.1 px sideways and adding 1.97 m/s of forecast speed; at point 12 it advances 0.19 [0.08, 0.29], more than the backward path (+6.55 px [4.87, 8.16]) and a norm-matched random edit (+2.54 px [1.16, 3.84]), but its sideways drift (6.1 px) is as large as the advance. The chord gives the same numbers as the spline (within 0.01 px). Reversing the time path flips the forecast heading on 0% / 7% of clips (points 12 / 22) against 99% for really reversed frames. So the time-step code is decodable and steerable in the encoder (held-out step within 0.098 / 0.106 steps, untrained 0.60 / 0.55) but the predictor does not use it as elapsed time; forecast timing comes from where the disk is in the frames, and the code behaves like a frame-index code amplified by training; the static-disk control (row above) supports a frame counter rather than an odometer. Patch instead of push[^tpatch]: on the same 128 carriers each context token's component in the rank-7 time subspace (the span of the 8 step centroids) is replaced by the same token's component from the same clip's full encoding two slots later, everything else untouched. At point 22 the forecast still does not advance (−0.17 [−0.32, −0.005] of two steps; pooled push −0.21 [−0.42, −0.01]; patch − push +0.03 [−0.18, 0.27]; real +2 window 1.15 [1.05, 1.24]), so the point-22 null is not an artefact of pooling. At point 12 the patch advances 0.34 [0.21, 0.49] (pooled 0.19 [0.08, 0.29]; patch − push +0.15 [0.005, 0.29]; patch − real window −0.81 [−0.96, −0.66]), so pooling was part of the problem there but most of the gap remains. At point 12 the advance tracks slot position rather than elapsed motion: on the real +2 window, patching in the time component of the same absolute frames taken from the full encoding, where they sat two slots later, advances the forecast 0.30 [0.17, 0.44], about as much as the +2 patch, while the same-slot patch on the original window gives 0.06 [−0.02, 0.14] and the backward patch −0.01 [−0.10, 0.07]; at point 22 the same-frames patch moves the forecast back instead (−0.27 [−0.44, −0.10]; same-slot −0.04, backward +0.05). Copying the whole later-slot tokens instead advances 0.84 [0.60, 1.11] (point 22) and 1.56 [1.32, 1.82] (point 12), so the forecast advances on the later slot's content outside the time subspace, not on the time code. The time patch also pushes the disk 15.5 / 8.4 px sideways (points 22 / 12), and reversing the slot order (slot 3 − s) flips the forecast heading on 23% / 4% of clips against 99% for really reversed frames. The random-subspace control is rank-matched but not size-matched (per-token change 9.7 against 63.1 at point 22), so the same-slot and same-frames controls above, not the random subspace, are the comparisons that count. The untrained copy was not run, because the position readers are fit on the trained predictor's forecasts. | `p5_time_predictor.json`[^timep], `p5_time_patch_predictor.json`[^tpatch] |
| Relational motion (two disks) | 637 two-disk clips balanced over common velocity c and relative velocity v_rel. A linear v_rel probe is by construction the difference of the two single-disk probes and gives no evidence of a relational code: V-JEPA 2 0.95 (point 2) vs untrained 0.98, and the V-JEPA readout fails the Galilean transfer across c halves (R² ≤ 0 vs 0.80 untrained). What training adds is identity-blind: relative speed |v1 − v2| reaches OOF R² 0.969 [0.964, 0.973] at point 17 (untrained best 0.204, pixels 0.00; `headline.best_by_target.abs_v_rel`), v_top − v_bottom 0.84, while v1, v2 and signed v_rel fall to 0.58–0.59 / 0.16 at the output; an MLP on activations beats the same MLP on the decoded single-disk velocities for |v1 − v2| from point 4 (+0.63 to +1.12 held-out R²). Per-disk pools contradict the paper's C.1.4 binding hypothesis in this setting: velocity is least bound to its own disk over points 6–17 (a plateau at 0.11–0.16, argmin point 9, vs 0.4–0.7 at point 1 and 0.21 at 20–24); measured against the other disk's tokens the untrained copy's index is ≈ 1, so training spreads one disk's velocity into the other disk's tokens; measured against the background instead (the object-vs-scene convention in §4.6) the untrained index is ≈ 0 at every depth while V-JEPA 2's rises with depth (0.12 at point 8, 0.39 at 22); that background pool decodes disk-1 velocity at R² ≥ 0.87 through point 9 in both models (mask leakage), so it measures how much disk signal the background keeps rather than binding, and the verdict on "bound" depends on the comparison set. Caveats: CIs resample clips not cells; the diskmask covers one disk; late-layer v1/v2 readouts are weak, which is what makes the composition null fail. Shape of the relative-speed code[^rshape] (knot = train folds 0–2, 306 clips; probe = folds 3–4, 203; 128 test clips scored only): the |v1 − v2| class means are ordered along their first PC (Spearman 0.96 at blocks 12–19, 0.89 at 22; 0.86–0.96 on held-out probe clips) but curved: the cross-validated share of centroid variance off a straight line is 0.39 / 0.41 / 0.44 / 0.41 at blocks 12 / 16 / 19 / 22 (0.05 at block 1; 0.80 for shuffled labels), while single-disk speed in the same clips sits at 0.74–0.82, the shuffle level, and the untrained copy at 0.75–0.77 (uncross-validated; its cross-validated values are unstable). On held-out test clips a linear read of |v1 − v2| from activations reaches R² 0.905 / 0.948 / 0.961 / 0.966, against 0.206 / 0.086 / 0.031 / 0.026 for a flexible quadratic in the decoded v1 and v2 (gap +0.70 [0.57, 0.85] at block 12, +0.94 [0.88, 1.02] at 22); the untrained copy is reversed (0.10–0.12 against 0.93–0.94). An encoder-level edit along the |v1 − v2| line moves the decoded |v1 − v2| 0.84–0.96 per +1 m/s step with v1 and v2 changes inside their intervals around 0 (largest leak ratio 0.12 / 0.08 / 0.09 / 0.03), while the positive control along the v1 line moves v1 0.33–0.71 and |v1 − v2| about 0. So in V-JEPA 2 |v1 − v2| is not recoverable from the decoded speeds: a residual relational component beyond what the object readers capture. Caveats: the quadratic baseline is fed linearly decoded v1 and v2 whose own test R² is 0.54–0.82 in V-JEPA 2 against 0.98 untrained, so the gap partly tracks decoder quality; the shuffle null is an uncross-validated residual (raw against raw still separates, 0.48–0.52 against 0.80) and the cross-validated values carry no interval; the leak test is weak (per-disk change intervals about ±0.2 m/s, the v1-line positive control moves v1 only 0.33 at block 22 with no interval, and a matched-norm random direction moves the per-disk readers by up to 0.33); clips, not design cells, are resampled; the residual is measured in the full 1024-d space. | `p5_relational_motion.json` (`headline`, `sources`, `beyond_composition`, `binding`), `p5_relational_stimuli_validation.json`, `fig_relational_motion.png`, `fig_relational_binding.png` |
| Temporal locality (which frames carry the code) | Direction and speed are readable almost equally from every frame pair: per-tubelet ridge CV R² spans 0.817–0.845 at point 1, 0.964–0.974 at 8, 0.979–0.987 at 12 and 0.986–0.990 at 22 (speed 0.949–0.952 at 1, 0.988–0.990 at 22). Because attention is bidirectional this says where the code can be read, not where it comes from, so frame groups were mean-ablated at point 12 (mean over 480 probe-role clips, 200 held-out clips): the predictor's forecast direction degrades more when the last context frames are removed (frames 7–8: 12.2°, +3.4° over the unablated 8.8°) than the middle (frames 3–6: +1.9°) or the first (frames 1–2: +0.3°), and fails only when all context is removed (87.6°, chance); at the encoder output any single frame pair costs ≤ 0.3° and the middle 8 frames 2.2°. By block 12 the direction code is spread over all frames with a mild recency weighting in what the predictor uses. Readers: forecast probe CV R² 0.907, encoder-output 0.985; bf16 vs fp32 forecast parity 3.2°. | `p5_temporal_locality.json` (`per_tubelet_probe.direction.<pt>.per_tubelet_cv_r2`, `ablation_point12.forecast.<cond>.{err_deg,err_increase_vs_none_deg}.mean`, `ablation_point12.encoder_output.*`, `readers.*.cv_r2`), `fig_temporal_locality.png` |
| Contact dynamics (a wall bounce) | The supplied clips have no contact, so 96 wall-bounce clips (elastic reflection at frame 3–12, speeds 2–4 m/s, a 6 px wall bar) and 96 straight twins were rendered with the validated twin renderer (features within 3e-4 of stored ones). Direction probes fit only on constant-velocity clips track the bounce frame by frame: the turn fraction (0 = incoming, 1 = reflected) goes from 0.22 before contact to 0.80 after at point 22 (0.17 → 0.63 at point 12; 0.24 → 0.80 at 25), crossing at the contact pair (0.53 at offset 0, 0.76 at +1), with 86% of post-contact frame pairs closer to the reflected direction; pre-contact pairs are already pulled ~0.2 toward the reflection (bidirectional attention mixing in future frames; the wall is a new object). The predictor does not anticipate the bounce: given frames 1–8 with the disk heading at a visible wall, the forecast keeps the incoming direction at every future step (turn fraction 0.17 / 0.09 / 0.07 / 0.00 at steps 0–3, ~22° from incoming, ~90° from reflected) while the encoder reads the reflection in the real future frames (0.86 / 0.75 / 0.53 / 0.67). Rendered stimuli (cross-render caveat); point 1 fails on wall clips. Two follow-ups[^cic]. (a) Bounce inside the context (96 bounce clips with contact at frames 2–6, 96 straight twins, 96 straight-out references): the forecast carries the observed reflection forward, turn fraction 0.84 [0.75, 0.93] averaged over the four forecast steps (0.89 / 0.90 / 0.77 / 0.82 at steps 0–3; contact at frames 2–3 1.00 [0.90, 1.10], at 5–6 0.66 [0.51, 0.80]), against −0.00 for the straight twins and 1.01 for the straight-out references; the context-only encoding at point 22 reads 0.72 [0.67, 0.76] after contact (last context pair 0.89 for early contact, 0.32 for late), and the untrained copy reads bounce clips near noise (0.15, interval spanning 0). The wall bar is drawn on bounce clips only, a small confound against the straight-out references. (b) Probe domain: a reader trained on the other set's bounces reads the in-context turned forecasts at 0.94 [0.87, 1.01] yet gives 0.07 [0.03, 0.12] on the stored set where the bounce falls in the forecast window, so the failure to anticipate is not a reader artefact; the most generous reader (all bounces, held-out clips) leaves 0.29 [0.22, 0.36], an upper bound since it may learn "wall present → reflect"; a probe-free nearest-future readout is uninformative (forecasts sit at about 0.5 between the bounce and straight futures for every clip type); and the 0.22 pre-contact "turn" at point 22 rises by 0.040 [0.037, 0.044] per visible post-contact frame with intercept −0.01 [−0.03, 0.02], i.e. future frames leaking through bidirectional attention. So the predictor carries an observed reflection forward but does not anticipate one. (c) An injected bounce direction makes it turn (0.48 [0.40, 0.55] on all tokens at block 22, 0.45 [0.40, 0.50] on the disk tokens at block 12), but a no-wall heading edit of the same norm turns it 0.77 [0.68, 0.86] and 0.46, so the edit writes the post-contact heading; the contact event is not shown steerable (§4.5). | `p5_contact_dynamics.json` (`results.per_point.<pt>.bounce.{pre_contact,straddle,post_contact}.turn_fraction.mean`, `results.per_point.<pt>.turn_fraction_by_offset`, `results.predictor_forecast.steps.step{s}_tubelet{t}.forecast_turn_fraction_post`, `results.real_future_encoder_turn_fraction_post_point25`, `results.parity_fp32_timepool_rel_maxabs_vs_stored`; provenance commit b8856e8 dirty, 645 GPU-s on box 53235298), `fig_contact_dynamics.png` |

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
  half-frame transfer and jumps back to chance, which the random network never does (§3.1). So the zone is a claim about the per-patch readout, which the paper's C.5 says, and a mean-pooled
  curve can neither confirm nor refute it. VideoMAE matches V-JEPA 2 on every pooled Part 1 measure and, from point 8, per patch (§3.1), so none of this is specific to latent prediction or shows that the variables are used to predict. The authors' OpenReview response states that "all 13 models encode motion direction
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
  only (13 on hard seeds 1–2), so their onsets of 4 and 6 are upper bounds. Speed was not run per patch.
- **Training dynamics.** With intermediate V-JEPA 2 checkpoints, the random-init vs final contrast becomes a curve.
  That is the natural test of when precision and the ring appear.
- **Predictor readout is a probe; the edit overshoots.** A probe of the pooled forecast, not a rendered future;
  point-22 edits overshoot in position (R 1.3–1.8); one stimulus, four targets in one 45° arc. The spline's lead is in
  angle only and the smoothing spline (the Part 2 default) is the worst real arm at the endpoint (over the path the chord is).
- **The late causal follow-ups are small.** The saddle and radial tests use 16 carriers on one held-out arc, one
  draw of 20 random axes or planes, and points 12 / 22 (radial: 22 only). Session 3 steers speed on one held-out block
  (128 carriers × 4 targets), and its speed edits change the forecast's speed code but not the forecast disk's motion.
  The twin-difference arm is one fit set (seed 0, 2 clips per value) at points 22 and 12 (point 12: chord comparison only). All four files were scored from a
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
- **Conceptor comparison is outside COAST's regime.** COAST's aperture rule found no aperture in its overlap
  band at either point, its pseudoinverse AND is invalid for 87–90% of the steered pairs (Jaeger's AND used instead),
  the target-aimed arm is my variant; the COAST-faithful rerun over 16 arcs (§4.4) is also a null, and its one working arm (uncentred, target-only, β = 1) is not COAST. The negative is for conceptor
  steering of a ring code, not for COAST on its success/failure task.
- **Off-target readouts are transferred probes.** The speed probe is fit on the speed set and is 0.56 / 0.44 m/s off on
  unedited direction clips; off-target changes are read as ratios to natural spread, and spline and chord are compared
  on unpaired intervals (§4.4).
- **New Part 2 files lack clean provenance.** The conceptor, off-target, FFT, density, held-out two-route and
  straightening scripts were uncommitted when these results were written; the density and conceptor files record a
  dirty worktree at 3c13095, the two-route files an unknown commit, and the FFT, off-target and straightening files no
  commit.
- **Session-2 twin-forecast cache does not match a recompute (unresolved).** The Fourier-4 predictor run recomputed
  the rendered twins' forecasts and they differ from the session-2 cache (relative L2 median 0.016, about 7% of the median twin-minus-source change, p90 0.082, max 0.31; the 0.995 in the same file is the positive control's forced choice, not an agreement figure;
  for 8% of recomputed twins the nearest cached twin belongs to a different target), although all 800 decoded twin clips
  hash-match and the source forecasts match the cache to 4e-4. That run uses its own recomputed forecasts as reference
  and keeps the cache-referenced numbers in its JSON as a sensitivity check. The twin ceilings quoted from the
  natural-norm files (9.1°, R 0.96) come from that cache (9.26° when recomputed) and are to be rechecked; the cause is not resolved here[^f4p].
- **The acceleration comparison is on our grid, not the paper's clips.** The magnitude result (§5) uses 240 rendered
  clips in which |a| is decorrelated from speed and displacement; the paper's own acceleration clips, which start from
  rest, were not tested, so the source of its early-layer number is inferred, not shown.
- **Not run from the paper's appendix:** the §6.3 / Tab. 2 local-attention masking test, C.6 (attention-distance analysis), C.7 (neuron direction tuning) and C.8 (neuron
  speed tuning). C.9 (feature dimensionality) is the nullspace analysis of §3.2.
- **Sample size vs d.** Around 1,200 train clips against d = 1,024 makes K a ridge count at a CV-chosen α. The K
  values should be compared across layers only at a fixed α (see the caveat in `p1b_*_dims.json`).

### 7.1 Sample sizes, units of resampling and detectable effects

One row per headline result. "Held out" names the clip roles: knot clips build the curve or basis, probe clips fit the
reader, test clips are steered and scored, and a held-out block of values is never used for knots. The last column is a
post-hoc minimum detectable effect, taken as roughly 2 × the bootstrap SE, i.e. the 95% half-width: a difference smaller
than it would not have been resolved by the design as run. It is read off the intervals after the fact and is not a
power calculation.

| result | unit of resampling | n | held out | CI method | headline estimate [95% CI] | half-width ≈ post-hoc MDE |
|---|---|---|---|---|---|---|
| Layer-wise probes (§3.1) | clip | direction 1,200 train / 300 test; speed and acceleration 1,228 / 308 | 5 CV folds inside train (grouped-fold rerun in `p1a_grouped_cv.json`); onset on out-of-fold predictions | fold SD for R² (no bootstrap); 200-draw clip bootstrap for onset | direction CV R² 0.980 ± 0.0011 SD (point 9), 0.9905 ± 0.0003 (peak, point 22); onset 2 [2, 2] | fold SD, not an SE: the 5 folds' training sets overlap by 75%, so 2 × SD/√5 ≈ 0.001 R² understates the uncertainty and is not an independent-sample interval; onset resolved to one sampled point |
| INLP (§3.2) | CV fold | 5 folds over 1,200 train clips; random-removal control 10 seeds | nested K on held-out folds | none (per-fold K reported) | K = 37 (fold range 33–42, mean 38.2); paper protocol 46 | no interval: ±4.5 probes is the spread of per-fold K, each fit on 4/5 of train, not an interval on the pooled K |
| Multi-probe steer (§3.3) | random basis draw (the null), not clips | 300 test clips; 200 null draws (p floor 0.005) | the evaluation probe is fit on the same 300 steered clips (C.12's protocol), so nothing is held out from the reader | permutation p against rank-2K and rank-matched nulls; no clip-level interval on the learned MAE | N = 10: 3.12° against rank-2K null 8.63° ± 4.18 SD (p 0.040); N = 1: 78.4° against 78.2° ± 2.45 (p 0.55) | not an MDE on the learned curve: the null's spread (2 SDs ≈ 8° at N = 10) |
| Endpoint bake-off, 6 arms (§4.3) | clip within arc (all a clip's targets together) | 16 arcs × 8 targets × 48 test clips | 45° arc of 8 values; probe folds for the reader | 1000-draw clip bootstrap; statistic = mean over arcs of per-arc means | FITPACK − raw chord +1.48° [1.27, 1.69] (point 12), +2.26° [2.09, 2.42] (point 22); interpolating − chord +0.81° [0.74, 0.89] (point 22) | 0.1–0.2° (paired); ≈ 1° at the chord's norm. Arc-level: see below |
| Predictor as judge, direction (§4.5) | carrier | 200 carriers × 4 targets | 45° arc; readers fit on unedited forecasts of probe clips | 1000-draw carrier bootstrap | spline − chord at the chord's norm −7.74° [−10.64, −4.98]; along the path −13.4° [−15.7, −11.0] | ≈ 2.3–2.8° |
| Disk-token sweep (§4.5) | carrier | 200 × 4 | as above | 1000-draw carrier bootstrap | source-disk-only 44.7° [38.8, 50.4]; source − union +3.8° [2.3, 5.3]; Fourier on source tokens − chord −7.7° [−10.6, −5.1] | 1.5° (paired) to 6° (arm means) |
| Natural-norm point 12, disk tokens, background control (§4.5) | carrier | 200 × 4 | as above | 1000-draw carrier bootstrap | chord 79.7° [72.5, 86.6] vs 92.1° unedited; spline − chord +8.1° [6.5, 9.7]; disk tokens 40.8° [35.6, 46.3]; background same-count − disk +48.0° [41.9, 54.6], energy-matched +39.3° [32.9, 46.0] | 1.6° (paired) to 7° (unpaired arm means) |
| Token patching (§4.6) | pair | 64 random + 64 position-matched pairs | reader probes on probe-role clips | pair bootstrap (draw count not recorded in the JSON) | disk share 0.88 [0.86, 0.90] (point 12), 0.22 [0.20, 0.24] (point 22), random pairs | ≈ 0.02 |
| Velocity sheet v2 (§4.4) | clip (all rows of a clip together) | 299 / 292 / 299 clips (2-block, 3-block, cross designs) | the target (direction, speed) cells | 1000-draw clip bootstrap | sheet − interpolating sequential, direction error −2.64° [−3.18, −2.14], −5.48° [−5.99, −4.98], −6.19° [−6.73, −5.62] | ≈ 0.5° |
| Velocity sheet through the predictor (§4.4) | carrier | 200 carriers (point 22, v1 block; a 48-carrier pilot agrees) | as v1 | 1000-draw clip bootstrap | sheet − interpolating sequential −4.8° [−6.2, −3.3]; sheet − interpolating ring +1.6° [0.1, 3.3] | ≈ 1.5° |
| Time through the predictor (§5) | clip | 128 of a 240-clip pool | held-out speed-set clips | clip bootstrap (draw count not recorded) | +2-step advance −0.21 [−0.42, −0.01] of two steps (point 22), 0.19 [0.08, 0.29] (point 12) | ≈ 0.1–0.2 of two steps |
| Time through the predictor, patch instead of push (§5) | clip | 128 | held-out speed-set clips (same carriers as the push run) | 1000-draw clip bootstrap (library default) | advance 0.34 [0.21, 0.49] (point 12), −0.17 [−0.32, −0.005] (point 22); patch − push +0.15 [0.005, 0.29] (point 12) | ≈ 0.14–0.16 of two steps |
| Time frame shuffle (§5) | clip | 308 test clips × 3 derangements × 8 slots | reader fit on unshuffled train clips | 1000-draw clip bootstrap | slot coefficient 0.956 [0.953, 0.958], content 0.002 [0.000, 0.004] (point 12) | ≈ 0.003 |
| Acceleration decorrelated from mean speed (§5) | clip | 240 rendered clips | ungrouped 5-fold CV (each clip its own group, so each test clip's design cell has about 10 training clips; displacement = mean speed × 0.625 s exactly, so 'beyond mean speed and displacement' means beyond mean speed) and leave-one-cell-out (20 folds; signed a mean-pooled 0.53 at point 12; Cartesian per-tubelet 0.08 [−0.12, 0.24] / 0.27 [0.02, 0.43] / 0.66 [0.52, 0.75] at points 9 / 12 / 22 against 0.16 / 0.27 / 0.62 per-clip, cell-block intervals). Ridge α is still chosen by an inner KFold that ignores cells, and leave-one-cell-out hides the (mean speed, a) combination while the same a value is seen at other mean speeds | clip bootstrap | mean-pooled R² beyond mean speed 0.44 [0.33, 0.53]; per-tubelet residual after an out-of-set speed decoder 0.32 [0.17, 0.44] (point 12) | ≈ 0.02–0.10 |
| Coordinate competition (§4.4; encoding result) | probe clip | 738 knot / 490 probe clips per point, 26 points × 2 copies | knot folds 0–2 fit, probe folds 3–4 score | 500-draw probe-clip bootstrap | polar − Cartesian (rank 2) +0.015 [0.009, 0.022] held-out R² (point 12) | ≈ 0.006–0.01 |
| Contact inside the context (§5) | clip | 96 bounce + 96 straight twins + 96 straight-out | forecast readers fit on 480 probe clips | clip bootstrap | step-mean turn fraction 0.84 [0.75, 0.93] | ≈ 0.09 |
| Contact probe domain (§5) | clip | 60 out-of-window bounce clips, 96 in-window | cross-set reader refits | clip bootstrap | transfer reader 0.07 [0.03, 0.12] out of window vs 0.94 in window | ≈ 0.05–0.07 |
| Contact steer and no-wall heading control (§4.5) | clip | 96 straight-twin carriers | leave-one-pair-out bounce direction; transfer reader never saw an in-context clip | clip bootstrap | bounce − no-wall heading edit −0.29 [−0.38, −0.21] (block 22, all tokens), −0.01 [−0.03, 0.01] (block 12, disk tokens) | ≈ 0.02–0.09 |
| Four-number edit through the predictor (§4.4) | carrier | 200 × 4 (extrapolation: 179 carriers) | 45° arc; half-ring folds for extrapolation | carrier bootstrap | Fourier − chord +2.9° [1.0, 4.5] (block 22, natural), −7.8° [−10.8, −5.1] (block 12 disk tokens) | ≈ 1.8–2.8° |
| Relative-speed shape (§5) | clip | 306 knot / 203 probe / 128 test | knot fit, probe CV, test scored only | clip bootstrap | activations − decoded-v1,v2 quadratic, held-out R² +0.70 [0.57, 0.85] (block 12), +0.94 [0.88, 1.02] (block 22) | ≈ 0.07–0.14 |
| Block-22 forecast without the heading probe (§4.5) | carrier | 200 × 4 | 45° arc | carrier bootstrap | recovery 0.186 (spline) / 0.234 (chord) / 0.112 (null); twin identification 0.331 / 0.312 / 0.249 (chance 0.25; real twin 0.92) | ≈ 0.02 |
| Acceleration magnitude |a| on the grid (§5) | clip | 240 rendered clips | 5 cell folds | 2000-draw clip bootstrap | partialled |a| R² −0.01 (block 12), 0.16 [0.04, 0.26] (block 22, mean-pooled); |a| − signed −0.88 [−0.92, −0.84] (block 12, per-tubelet) | ≈ 0.04–0.11 |
| Twin-free disk-token dose (§4.5) | carrier | 200 × 4; dose profiles from 480 probe-split clips (7–8 per target angle) | carriers never in the probe split | carrier bootstrap | twin-free − per-token +4.9° [3.7, 6.3]; 77% [70%, 82%] of the uniform-to-per-token gap recovered | ≈ 1.3° (paired) |
| Acceleration grid, cell-level bootstrap and power (§5) | design cell (20 cells × 12 clips) | 240 clips | 5-fold; leave-one-cell-out; leave-one-mean-speed-level-out | 20-cell block bootstrap; 200-permutation null within mean-speed level | signed a mean-pooled 0.44 [0.17, 0.56] (point 12; ours [0.12, 0.55]); |a| 0.17 [−0.21, 0.35] (point 22; ours 0.16 [−0.24, 0.36]); null p95 0.002–0.009; cell-block intervals are conservative (both targets constant within a cell) | ≈ 0.2 (cell bootstrap); planted-signal detection 90% at oracle R² 0.066 |
| Probe-free readouts at matched size (§4.5) | carrier | 200 × 4 | 45° arc; readers as the Fourier-4 run | 1000-draw carrier bootstrap, seed 0 | block 22 spline − chord: forced choice −0.035 [−0.052, −0.019], recovery −0.048 [−0.057, −0.040], twin identification +0.019 [+0.001, +0.037] | ≈ 0.01–0.02 |
| Model size, layer curves (§3.5) | clip | direction 1,200 / 300, speed and acceleration 1,228 / 308 per model; ViT-L / ViT-H / ViT-g, trained and untrained | 5-fold CV inside train; test read once | 200-draw clip bootstrap (onset), 1000-draw (test R²) | direction onset block 2 / 3 / 4 (0.083 / 0.094 / 0.10 of depth), each [x, x] | one sampled point (onset; clip resampling only; onsets 0.125 / 0.125 / 0.10 at a 95% threshold); zone shift earlier by first recovery (0.375 → 0.28) and later by durable recovery (0.375 → 0.44); peak block not stable (argmax 0.56–0.95 of depth on a flat top); hard-render zone readout ViT-H only, seed 0 |
| Head ablation (§4.7) | clip | Part A 300 train clips; Part B 300 direction / 154 speed test clips | probe fit on train, scored on test | clip bootstrap; four count-matched random head sets | eight top heads cost 0.040 [0.035, 0.045] of point-12 direction R² against 0.124–0.220 for random sets (a null); tracking heads 19 (weak criterion) / 3 (strict, none in the ablated blocks); point 22 0.0051 [0.0037, 0.0066]; untrained Part B invalid (bf16 baseline −15.7 against 0.876 fp32) | ≈ 0.005 (joint drop); random-set spread 0.10 |
| Static-disk control (§5) | clip, paired | 96 static + their 96 moving twins | held-out speed clips re-rendered | 1000-draw clip bootstrap, ratio of means | static ÷ moving 1.07 [1.05, 1.08] (mean direction, point 12) | ≈ 0.02–0.04 |
| Acceleration through the predictor (§4.5) | carrier | 128 × 4 targets | held-out 7.52–8.61 m/s² block | 1000-draw clip bootstrap (SE stored) | spline error 1.82 [1.65, 2.00] m/s² (point 21, own norm); spline − chord −0.010 [−0.012, −0.007] | ≈ 0.18 m/s²; 0.003 paired |
| COAST as written (§4.4) | clip within arc, per arc | 16 arcs × 384 steers | 45° arc | per-arc clip bootstrap; summary mean ± SD over arcs | every conceptor arm 91.0–102.0° against the chord's 6.7° / 4.2°, CI above zero on 16/16 arcs | effect ≫ interval |
| Energy geodesic (§4.4) | carrier, per target | **headline arc only**, 8 targets × 16 carriers per point; 0 of 56 + 56 solves converged | 45° arc | per-target CIs | kNN geodesic − chord min radius +0.08 (point 12, 8/8 target CIs above 0), +0.02 (point 22, not all) | single arc; not resampled over arcs |

Two limits of the intervals. First, the Part 2 clip bootstrap conditions on the 16 arcs drawn: it resamples clips inside
each arc, so arc-to-arc variation is not in the interval. From the unified run's per-arc means, the arc-level SD of
the FITPACK − raw chord endpoint gap is 1.98° (point 12) and 1.78° (point 22), a paired arc-level half-width (2 SD / √16)
of 0.99° and 0.89°, about five times the clip-bootstrap one; the gaps (+1.48°, +2.26°) still exceed it. The paper's
interpolating spline at point 22 is +0.81° ± 0.65 and excludes zero; at point 12 on the label-free angle it is +27.9° ±
21.5 (SD 42.9, three arcs carry the mean). The Fourier edit clears its arc-level interval at both points (−1.90° ± 0.80,
better on 15/16 arcs; −0.53° ± 0.20, 15/16), the Reinsch smoother at point 22 (−0.45° ± 0.11, 16/16) but not at point 12
(−0.38° ± 0.70, 7/16), and the probe-subspace steer at point 12 (−2.53° ± 0.87, 16/16) but not at point 22 (+0.04° ±
0.70). Second, the designs in the table are not independent samples of the network: they share one stimulus family, one split
and one encoder.

**Two thin spots.** (1) The velocity sheet through the predictor was a 48-carrier pilot; the 200-carrier rerun now used
in the text keeps every sign and makes both directions of the comparison significant: the sheet beats composed 1-D edits
on direction (−4.8° [−6.2, −3.3] against the interpolating sequential edit) and loses to a direction-only ring edit on
direction on the pooled probe (+1.6° [0.1, 3.3]) but beats it on the disk-position heading (−7.1° [−9.3, −5.0]) and on speed. (2) Results still quoted from the
single headline arc: the energy geodesic, the conceptor fit of the first COAST run, the point-12 edit's attention/MLP
repair attribution (16 carriers), the saddle and radial tests (§7), and the headline-arc columns of the §4.3 table. Of
the conclusions that rest on them, the geodesic "kNN geodesic sits above the chord at point 12" excludes zero per target
on that one arc but is untested over arcs; the COAST null is replicated over 16 arcs; and every endpoint comparison in
§1 and §4.3 now quotes the 16-arc number beside the headline arc's.

Planned in the spec and not run: the last-frame-only control of the layer curves (a direction floor is expected, since one frame carries position, not motion); it is disclosed here rather than in the tables.

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
  Exceptions: `results/p5_velocity_sheet.json` (v1) records `commit: null` and `git_commit: "unknown"` (run on a box
  at 18:45 UTC from `/workspace/wm_p5`); no log names its commit, so it is unknown (the file was first committed in
  42b30fe). A scan of every results JSON (18:45 ET) finds 53 files that record `commit: null` with no usable
  `git_commit` (null or "unknown"): the 32 per-arc conceptor files `results/arcs_conceptor16/L{12,22}_s{1..16}/`, the 8
  misaim-fix raw-chord arcs `results/arcs_rawchord_fixed/L12_s{1,4,5,7,8,9,11,13}/`, the four geodesic files
  `p2_geodesic_direction_L{12,22}{,_full}.json`, `p2_two_route_heldout_L{12,22}.json`, the time files
  `p5_time_manifold.json`, `p5_time_manifold_controls{,_subset768}.json`, `p5_time_positional{,_subset768}.json` and
  `p5_time_static_control.json`, and `p5_velocity_sheet.json`; their code state is known only from file mtimes and the
  session logs. The v2 sheet, acceleration sheet, sheet-through-predictor, motion-geometry, untrained token-patching,
  natural-norm direction and time-predictor files record d97502d with dirty scripts, and the repair attribution 3c13095
  dirty.
- **Numerics.** CPU–GPU parity on 8 clips: worst per-layer max|Δ|/max|x| 8.2e-5 (rule < 1e-3); GPU batch-8 vs
  batch-16 gap 1.31× the CPU–CPU gap (rule ≤ 2×). Frame hashes and disk masks match, and 27/27 sha256 checks of the
  downloaded activations pass[^gpu].
- **Cost.** GPU session 1 (RTX 4080 SUPER, Vast): 37.7 billed minutes, $0.19[^gpu]. Session 2 box (RTX 4060 Ti,
  $0.198/h): $0.512 to the end of session 2 (the run itself $0.311); $1.19 over 6.00 billed hours as of 22:44 ET on 27 September
  (the file's last snapshot; the box was live at that time), including the native-readout extraction (0.09 h, $0.018), the norm-matched rerun (0.137 h, $0.027),
  the along-path forward (0.34 h, $0.067) and the reverse test (0.365 h, $0.072); $0.006 egress[^cost]. The per-patch
  run on the same box took 1.65 h ($0.33; box totals not refreshed since), and the encoder-output forward 817 s of GPU
  time with no separate cost recorded (`session2_encoder_output.json`, `forward.seconds_total`); the Goodfire-recipe
  pullback ≈ 62 GPU-min and the context-only extraction 2.0 GPU-min on the same box at $0.198/h (≈ $0.21, not in the
  $1.19 total). No other box's cost is recorded.
- **In flight at submission time:** none; every experiment launched has landed and is reported above.
- **Late inclusion.** `results/p5_motion_heads.json` (§4.7) landed at 21:01 ET, one minute after the stated 21:00 inclusion
  cutoff, and was included.
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
[^cost]: `results/session2_cost.json` (`cost_start_to_session2_done_usd`, `session2_run_cost_usd`, `cost_so_far_usd`, `billed_hours_so_far`, `as_of_utc` 02:44 UTC, `runs[0..3]`, `pull_egress_usd`; the file's `status` field reads "running" because it was last written at 02:44 UTC on 28 September while the box was live).
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
[^twd]: `results/session2_twin_difference_L22.json` (`fit_set.{n_fit_clips,n_fit_pairs,k_nn,svd_energy_top_r}`, `per_condition.{chord_norm,natural_twin_norm}.{twin_r1,twin_r2,twin_r4,twin_r8,twin_full,spline,chord,probe_qr}.{dir_err_to_target,R_dir_real_change,px_err_to_twin_true,enc_out_err_to_target,nearest_real_R_pt25,R_act_twin_pt25}`, `twin_minus_best_comparison.*` (best comparison = `spline`, the interpolating spline of the norm-matched rerun; best twin = `twin_full`), `unedited.dir_err_to_target`, `parity`); predictor readouts of the comparison arms from the norm-matched cache (`definitions."comparison arms"`); `scripts/session2_twin_difference.py`; 987 GPU-s (`forward.total_seconds`); scored at 3c13095 with `git_dirty_src_or_scripts: true`. Point 12: `results/session2_twin_difference_L12.json` (`per_condition.{chord_norm,natural_twin_norm}.twin_{r1,r2,r4,r8,full}.dir_err_to_target`, `twin_minus_raw_chord_chord_norm.<arm>.dir_err_to_target`, `predictor_comparison_own_norm_session2_cache.raw_chord_forwarded_here.dir_err_to_target`, `.arms.{spline,probe_qr}.dir_err_to_target`, `.norm_over_chord_median`, `twin_minus_best_comparison.{chord_norm,natural_twin_norm}.enc_out_err_to_target` (best comparison `probe_qr`, best twin `twin_r1`), `per_condition.*.*.enc_out_err_to_target`, `unedited.dir_err_to_target`; `forward.pred_compare` false); forward on box 2 (RTX 4060 Ti, 2,382 s, `forward.total_seconds`) with the pre-fix script, identical to the L22 code; scorer fixed for the missing point-12 comparison cache (`scripts/session2_twin_difference.py`, uncommitted); scored at ea65b26 with `git_dirty_src_or_scripts: true`.
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
[^geo]: Full run: `results/p2_geodesic_direction_L{12,22}_full.json` (`convergence.{batched_solves,converged}` 56 + 56, 0; `headline.pooled_arms_mean.{chord,spline,geo_knn_from_chord,geo_kde_from_chord}.{probe_radius_min,behaviour_energy,excess_to_nearest_real,LG_knn,LG_kde,ring_plane_frac_of_bend,bend_cos_with_spline,probe_err_to_target}`; `headline.pooled_paired_gaps.geo_{knn,kde}_from_chord_minus_{chord,spline}.probe_radius_min.{mean,per_target_ci95,all_targets_ci_excludes_0_same_sign}`; `headline.per_target[*].restart_null.{knn,kde}.{dist_to_geo_from_chord_mean,frac_restarts_LG_below_geo_from_chord_by_1pct}`, averaged over the 8 targets; `provenance.run_config` 48 clips per target, 8 targets, 3 restarts on 8 clips; `headline.pooled_note` still says 16 carriers and `run_config.lite` is true, both stale labels); the spline length ratio is `LG_knn`, `LG_kde` of `spline` over `chord`; the stored point-12 spline endpoint 9.73° is `results/p2_steer_direction_direction_L12_contiguous_rawchord.json` `summary.manifold.overall.probe_err_to_target`; scipy versions from `results/GEODESIC_NOTES.md` only. Lite run: `results/p2_geodesic_direction_L{12,22}.json` (`convergence` 1 of 21 and 0 of 21). `figures/fig_geodesic_direction_L{12,22}_full.png`; `scripts/run_geodesic.py` (`length`, `g_sqrt`), `tests/test_geodesic.py`. No git commit recorded; full files written 17:33 ET.
[^mak]: `results/p5_makelov_ranking_L22.json` (`spearman_inlp_order_vs_move.{angle_move,by_round_angle_move,rel_l2_move}.{rho,ci95}`, `extension_all_planned_directions`, `controls.{inlp_mean_move_deg,random_mean_move_deg,rawchord_move_deg,frac_inlp_dirs_above_random_p95,inlp_mean_rel_l2,random_mean_rel_l2,rawchord_rel_l2}`, `top5_by_move`, `bottom5_by_move`); `scripts/run_makelov_ranking.py`, `src/wm/makelov.py`, `tests/test_makelov.py`; plan built at 4bbd141, scored at fb3c9cb, dirty tree both times; 185 GPU-s on box 53235298 at 17:02 ET; frames 1–8, edit added to every token at point 22; bootstrap CIs resample the 16 carriers only.
[^c16]: `results/p2_conceptor_direction_16arc_L{12,22}.json` (`arms.<arm>.endpoint_err_deg.{mean,ci95}`, `arms.<arm>.gap_vs_raw_chord_deg`, `arms.<arm>.n_arcs_better_than_raw_chord`, `stored_16arc.{legacy_coord_aim,aim_fixed_on_8_misaimed_arcs}`, `reproduction_check`, `provenance.arc_files`); per-arc files `results/arcs_conceptor16/L{12,22}_s{1..16}/`; `scripts/aggregate_conceptor_16arc.py`, `scripts/run_conceptor.py --aim arc --lite` (4 random-projector draws per arc, not aggregated); point 12 default angle rule, point 22 `--labels-angle`; CPU on box 53030966, 17:13–17:18 ET; aggregate computed at c00e48e on a dirty tree, per-arc files record no commit.
[^epd]: `results/p2_endpoint_diagnosis.json` (`decomposition.<config>.{headline_seed0,arcs16}` for configs L22_labels, L12_unsup, L12_labels; `paper_vs_ours_steps` with causalab file:line; `foreign_knot_split`; `verdict`; `exploratory_note`); per-arc outputs `results/endpoint_diagnosis_raw/`; `figures/fig_p2_endpoint_diagnosis.png`; `scripts/run_endpoint_diagnosis.py`, `scripts/summarize_endpoint_diagnosis.py`; CPU on the Mac, 16:58–17:25 ET; reproduces the stored raw-chord arc files to 3e-13° on the endpoint gap.
[^vmpp]: `results/p1a_perpatch_direction_videomae.json` (`points` [0,1,2,4,5,6,8,9,12,16,22,24], `curves.{perpos_mean_r2,pooled_mean_r2,cross_half_r2,meanpool_r2,perpos_frac_ge_0.5}`, `onsets.*.onset`, `layers[].halves.cross_mae_mean`); test split 1,200 / 300; `scripts/p1a_perpatch.py` (VideoMAE path, uncommitted at run time, base b8856e8); `figures/fig1k_perpatch_videomae.png`; GPU on box 53030966, 16:55–17:21 ET.
[^bake16]: `results/p2_bakeoff_unified_16arc.json` (commit 98ca808, dirty; `scripts/run_bakeoff_unified_16arc.py`; `table.<arm>.{err_end,R_end_test,radius_min,radius_mean,monotone_frac,delta_norm_end}.{12,22}`, `paired_vs_chord.<arm> - chord_raw.*`, `extra.{12,22}.{radius_wp_mean_over_arcs,lambda_cv_per_arc,probe_qr_n_probes_per_arc}`, `headline_arc`); path stage and headline reproduction `results/p2_endpoint_diagnosis_path.json` (`scripts/run_endpoint_diagnosis_path.py`; `headline_reproduction.{stored,recomputed_K50,recomputed_K11}`, `aggregate_16arc`, `radius_along_path_K11_mean_over_arcs`); raw per-arc outputs `results/endpoint_diagnosis_raw/{unified,path}_L{12,22}_{a,b}.*`.
[^tstatic]: `results/p5_time_static_control.json` (`models.{vjepa2,random}.{12,22}.{static,moving_twin,original_stored}.{slope_mean_direction,slope_spline_coordinate,slope_mid_band_time_probe,mean_coord_per_step_spline,shared_curve_share_pca64}`, `static_over_moving_twin.{dir,spline,probe}.{ratio_of_means,ci95}`, `distance_to_moving_clips.clip_mean_feature_nn_dist.static_over_moving_twin`, `render.static_smoke`; 96 + 96 clips, clip bootstrap seed 0). Provenance records no commit (`git_commit: unknown`, run from a box copy at `/workspace/wm_p5`); the file's `verdict` string says the untrained CIs all include 1, but two of six (spline at point 12, probe at point 22) exclude it.
[^tpatch]: `results/p5_time_patch_predictor.json` (`scripts/time_patch_predictor.py`, `tests/test_time_patch_predictor.py`; commit 98ca808, dirty; 200 GPU-s on box 2, RTX 4060 Ti; `arms.{22,12}.{tp+2,tp_same,rp+2,tok+2,tp_rev,w2:tp-2,w2:tp_same}.{advance_fraction,across_abs_px,frac_heading_flipped_gt90}`, `gaps.<pt>.tp+2_minus_{pooled:sp+2,win+2}_advance_fraction`, `pooled_push_same_clips`, `references.win+2`, `edit_size.<pt>.<arm>.per_token_delta_norm_mean`, `design.patch_plan.rank` 7). 128 carriers; 95% clip bootstrap, 1000 draws (`wm.timeman.cluster_bootstrap` default; the draw count is not written to the JSON). Pooled-push arms re-read on the same clips reproduce `p5_time_predictor.json`.
[^coord]: `results/p5_coordinate_competition.json` (`scripts/run_coordinate_competition.py`, `tests/test_coordinate_competition.py`, `figures/fig_coordinate_competition.png`; box 2 CPU; run with script sha c9e600d0 (the pre-registration hash) and assembled on the Mac with a modified script (9e31b8d5…) that adds the post-hoc steering block and the exp4 note without changing the pre-registration text; the JSON's recorded commit 8ee1deb is wrong, since the script first entered git at 412df8f; the design was written after exp4 and the DFT were known, so it is pre-registered for this run only; `preregistration` (written 18:31 ET, `preregistration_script_sha256_at_writing` c9e600d0…), `winner_by_point.{vjepa2,random}`, `encoding.<model>.<pt>.{primary_speedset,transfer_speed_to_dirset}.{r2,per_candidate.<c>.minus_pca_k,h2h_rank2_polar2_minus_cartesian}`, `steering.{12,22}.arms`, `steering_posthoc_velocity_carriers` (labelled post hoc in the file), `existing_exp4_selection_note`). 500-draw probe-clip bootstrap; 20 random-feature draws. The superseded search is `results/p5_motion_geometry.json` `exp4_coordinate_search_and_exp5_shortcuts`.
[^vsp]: `results/p5_velocity_sheet_predictor_n200.json` (commit 8ee1deb, dirty; 804 GPU-s on box 4; parity to the native forecast cache 9e-7; `predictor.{own,sheet_norm}.<arm>.{dir_err_deg,R_speed_probe,R_disp_speed}`, `paired.<cond>.sheet_minus_<arm>_{dir_err,speed_err}`, `reference.{unedited,twin_forecast}.dir_err_deg`); 200 carriers, 1000-draw clip bootstrap, same seed and design as the 48-carrier pilot `results/p5_velocity_sheet_predictor.json` (sheet 27.6° [23.3, 32.3]; paired −5.8° / −3.9° / −4.9°). No interval that excluded zero in the pilot changes sign; one near-zero speed gap flips sign within its interval (sheet − sequential speed error −0.002 → +0.016), and one gap becomes significant against the sheet (own-norm sheet − sequential displacement-speed error +0.098 [0.015, 0.184]). Disk-position heading: `paired.<cond>.sheet_minus_ring1d{,_interp}_disp_heading_err`.
[^tshuf]: `results/p5_time_shuffle.json` (`scripts/run_time_shuffle.py`, `tests/test_time_shuffle.py`, `figures/fig_time_shuffle.png`; commit 412df8f, dirty; 875 GPU-s on box 3; seeds perms 0 / bootstrap 0; `models.{vjepa2,random}.<pt>.{unshuffled_test.t_r2,shuffled_pooled.{coef_slot,coef_content,frac_nearer_slot},shuffled_random_derangements,static.slope_decoded_on_slot}`); 308 test clips × 3 derangements × 8 slots; 1000-draw clip bootstrap.
[^adec]: `results/p5_accel_decorrelated.json` (`scripts/run_accel_grid.py`, `tests/test_accel_grid.py`, `figures/fig_accel_decorrelated.png`; commit 412df8f, dirty; 574 GPU-s on box 5; renderer parity IoU 0.984; ridge λ by 5-fold CV; `design_correlations_with_a`, `models.{vjepa2,random}.points.<pt>.{meanpool,timepool}.{ridge_a,ridge_a_partial_mean_speed_displacement,ridge_a_partial_decoded_speed_seq}`, `linear_a_from_decoded_speed_seq.inset_nested`, `decoded_speed.inset_nested.step_speed_r2`, `mlp.{act_meanpool,act_timepool,speedseq_inset_nested}.{r2,minus_other}`, `reference_mlps`); 240 clips, 5 clip folds (each clip its own group, cells spread over folds), 95% clip bootstrap.
[^cic]: `results/p5_contact_in_context.json` (`scripts/run_contact_in_context.py`, `tests/test_contact_in_context.py`, `figures/fig_contact_in_context.png`; commit 412df8f, dirty; 230 GPU-s on box 4; `results.forecast.{steps.<s>.bounce.all.turn_fraction,step_mean_turn_fraction.{bounce,straight_in,straight_out}.{all,early_kb2-3,late_kb5-6}}`, `results.encoder.{ctx8,random_enc16}.22.{bounce_context_by_phase,bounce_last_context_tubelet_t3}`; the recorded `script_sha256` (Mac scoring) and `forward_script_sha256` (box) differ because scoring was edited after launch) and `results/p5_contact_probe_domain.json` (`scripts/contact_probe_domain.py`, `figures/fig_contact_probe_domain.png`; CPU; commit 4c7ed91, dirty; `results.a_reader_refits.{native,+straight,+all_xfit,transfer}.{out_of_window_k_b>=8.clip_mean_post_steps,in_window.clip_mean_steps}`, `results.b_probe_free`, `results.c_pre_contact_pull.pooled_linear_fit_tf_vs_visible_post_frames`). 96 clips per kind; clip bootstrap (cells for the pre-contact fit).
[^dsweep]: `results/session2_disk_token_sweep.json` (`scripts/session2_disk_token_sweep.py`, `tests/test_disk_token_sweep.py`, `figures/fig_disk_token_sweep.png`; commit 1aa2eed, dirty; script sha 0d29ed7c…; 1,283 GPU-s on boxes 2 / 4 / 5; `table.<arm>.{err_to_target,R,edited_tokens}`, `paired.*`, `plan.own_norm_median`, `union_reproduction` (the stored 40.8° reproduced to 0.0° per clip), `plan.parity_*` (spline and Fourier deltas match the stored ones to 3e-8)). 200 carriers × 4 targets; carrier bootstrap, 1000 draws.
[^csteer]: `results/p5_contact_steer.json` (`scripts/run_contact_steer.py`, `tests/test_contact_steer.py`, `figures/fig_contact_steer.png`; commit b0cd652, dirty; box script sha c5a69a70…, Mac scoring sha f1ab119c…; box 4, a 23 s smoke and a 69 s corrected run; a first 75 s run was discarded for a disk-mask bug; `results.points.{22,12}.conditions.{none,a_pool_loo,b_disk_loo,c_rand_pool,c_rand_disk,d_far_pool,d_far_disk,e_own_pool,e_own_disk,e_own_tokens}.{forecast_turn_fraction_step_mean.{all,early_kb2-3,late_kb5-6},encoder_ctx_post_tubelets_turn_fraction_point25}`, `results.references`, `parity_none_vs_in_context_forecast_rel_maxabs` 0.0). 96 carriers; clip bootstrap.
[^rshape]: `results/p5_relational_shape.json` (`scripts/run_relational_shape.py`, `figures/fig_relational_shape.png`; git_commit 732010a (the `commit` field is empty); `roles.n`, `sources.{vjepa2,random}.<pt>.{q1_line.{abs_v_rel,abs_v1}.{spearman_pc1,probe_spearman_on_knot_pc1,cv_resid_frac,resid_frac,resid_frac_shuffle_null_mean},q2.heldout.test.{a_direct_r2,b_flex_decoded_r2,a_minus_b_flex},q3.{abs_v_rel_line,v1_line_positive_control,leak_frac_max_abs_dv_over_d_abs_v_rel}}`, `caveats`).
[^f4p]: `results/session2_fourier4_predictor.json` (`scripts/session2_fourier4_predictor.py`, `figures/fig_fourier4_predictor.png`; commit 48e17ed; 200 carriers × 4 targets; `task1.{table,paired}.{P22,D12}.{own,natural}`, `task1.positive_control_twin_forecast_session2_cache_vs_rerun_reference`, `task2.{table,paired}` (179 carriers in the half-ring folds), `encoder_level.{12,22}`, `parity`; `fitpack_arm` not computed). The `_smoke` files at neighbouring paths are a 4-carrier smoke run and carry no reported numbers. The spline-tangent extrapolation uses natural boundary conditions and linear extrapolation along the end derivative, as the authors' code does.

[^amag]: `results/p5_accel_decorrelated_magnitude.json` (`scripts/run_accel_grid.py`, `magnitude` stage; commit ecec750; features `artifacts/accel_grid/feat_{vjepa2,random}.npz` sha-matched to the signed run, whose partialled signed R² it reproduces at all 28 cells; `abs_a_levels`, `design_correlations`, `models.{vjepa2,random}.points.<pt>.{meanpool,timepool}.{abs_a,signed_a}.ridge_partial_mean_speed_displacement`, `.abs_a_from_abs_signed_ridge`, `.abs_minus_signed_partial`, `.mlp`). 240 clips, 5 cell folds (seed 0), 2000-draw clip bootstrap.
[^chead]: `results/p5_contact_steer_heading_control.json` (commit af17ffc; 65.5 GPU-s; `headline.{22,12}.{bounce_edit,no_wall_heading_edit_norm_matched,paired_bounce_minus_heading}`, `results.points.<pt>.{forecast_turn_fraction_step_mean,paired,ratio_heading_over_bounce_edit_effect}`; stored-condition reruns reproduce `p5_contact_steer.json` to 0.0). 96 carriers; clip bootstrap.
[^tfree]: `results/session2_disk_token_sweep_twinfree.json` (`scripts/session2_disk_token_sweep_twinfree.py`, `tests/test_disk_token_sweep_twinfree.py`; commit 3025adc, dirty; `definitions.doc`, `table.src_chord_{1x,uniform,twinfree_bin,twinfree_all,twinfree_bin_raw}.{err_to_target,R,per_token_dose_mean_median}`, `paired.*`, `frac_of_uniform_to_pertoken_gap_recovered.{bin,all}`, `parity_rerun_vs_stored_sweep` (per-token and uniform arms rerun to 1e-6 relative)). 200 carriers × 4 targets, point 12, source-disk token set; carrier bootstrap.
[^aaudit]: Independent acceleration audit (data-QA pass, re-derived from the mp4 pixels and the feature files, checksums matched; outputs in the session scratchpad `qa_accel/`: `s2_supplied_probes.json` (supplied-set probes with pixel-measured mean speed and displacement partialled; 5-fold mean ± SD), `s3_grid.json` (grid design correlations; signed / |a| / leave-one-mean-speed-level-out ridge R² with clip and 20-cell block bootstrap intervals), `s3_perm_*.json` (200-permutation null within mean-speed level; 40 for the per-tubelet point 1), `s5_mde.json` (planted-signal detection)). Not committed to `results/`.
[^acart]: `results/p5_accel_decorrelated_cartesian_loco.json` (commit 52d6f01; `models.{vjepa2,random}.points.<pt>.{meanpool,timepool}.{per_clip_folds,leave_one_cell_out}.{signed_a,abs_a,cartesian.{components,mean}}`, `folds.leave_one_cell_out` (20 folds, all 12 clips of one cell held out), `design_correlations`); mean speed and displacement partialled inside folds; 2,000-draw clip bootstrap.

[^pfms]: `results/session2_probefree_matched_size.json` (`scripts/session2_probefree_matched_size.py`, `tests/test_probefree_matched_size.py`, 2 tests pass; commit bd572f4, dirty; `matched.{P22_natural,D12_natural_union,D12_natural_src}.{spline,chord,spline_minus_chord}.{forced_choice_frac,recovery_full,twin_id_acc,heading_err_deg}`, `matched.*.edit_size`, `ladder.{P22,D12}.{spline_own,chord_interp_at_spline_size,spline_minus_chord_all_pairs,spline_minus_chord_in_bracket_pairs,caveat}`, `parity` (matched arms reproduce the Fourier-4 table to 0.0); the block-12 union spline dose in `edit_size` is copied from the chord's `union_chord_1x` run, equal by construction but not measured on the spline arm). 200 carriers × 4 targets; carrier bootstrap, 1000 draws, seed 0; reference twin forecasts from the Fourier-4 rerun.
[^acell]: `results/p5_accel_decorrelated_cellboot.json` (commit 379cf72, dirty; `models.{vjepa2,random}.points.<pt>.{meanpool,timepool}.{signed_a,abs_a}.ridge_partial_mean_speed_displacement.{r2,ci95_clip,ci95_cellblock}`; the refit out-of-fold predictions reproduce every R² in the magnitude file; `keys.not_covered`: plain ridge, |a|-via-signed, MLP and decoded-speed readouts). 1,000 draws over the 20 cells, seed 0. The magnitude file's recorded `script_sha256` (f4189407…) predates the Cartesian / leave-one-cell-out edits to `scripts/run_accel_grid.py` (commit 4d766b3), so it no longer matches the script on disk (c11de40f…); the analysis code paths it used are unchanged.
[^scale]: `results/p5_scaling_layerwise.json` (`scripts/run_scaling_layerwise.py`, `figures/fig_scaling_layerwise.png`; committed in c1a0f36, file provenance `git_commit` c1a0f363; `models.{vitl,vith,vitg}_{pretrained,random}.variables.<var>.{layers[].{point,frac,cv_mean,test_r2,test_r2_ci95},summary_cv.{onset,onset_ci,onset_frac,peak,peak_score}}`, `zone_halfframe_hard.{vitl,vith}.{points,cross_half_r2,summary}`, `vitl_crosscheck_vs_part1_files` (maximum difference 1.7e-13), `parity_vitl_bf16`, `extract_info` (ViT-H from `facebook/vjepa2-vith-fpc64-256`), `definitions`). 1000-draw test-clip bootstrap for test R², 200-draw train-clip bootstrap for onsets; final merge commit 4b3c562 includes the untrained ViT-g acceleration / (ax, ay) curves; `extract_info.*.total_gpu_seconds` sum to 4,572 s, `zone_halfframe_hard.vith.extract_gpu_seconds` 92 s; `provenance.notes`: the script was resynced mid-run with extraction and probe maths unchanged.
[^heads]: `results/p5_motion_heads.json` (`scripts/run_motion_heads.py`, `tests/test_motion_heads.py`; file provenance commit 89c7a30, dirty, script sha e9c7f5b8… on the Mac and c093c391… on the box (scoring and figure code only); 2,924 GPU-s; `part_A.{vjepa2,random}.{table,block_median_ratio,block_max_ratio,block_median_phantom_ratio,n_heads_ratio_gt_2,n_heads_labelled_tracking,tracking_heads}`, `part_B.<model>.summary.{direction,speed}_p{12,22}.{<block>,global}.{max_head,max_head_drop,topk_joint_drop,random_k_drops,top_joint_drop,random_matched_drops}`, `framing_and_caveats`). Part A on 300 train clips (fp32 eager attention), Part B on 300 direction / 154 speed test clips (bf16 autocast, drops relative to the same pipeline's unablated baseline, test R² 0.988 against 0.9885 stored); clip bootstrap.
[^epdnorm]: `results/p2_endpoint_diagnosis.json` `configs.{L22_labels,L12_labels}.arcs16.arms.{add:spline_smooth,add_norm:chord_raw@spline_smooth,add_norm:spline_smooth@chord_raw,add:chord_raw,add:oracle,add:causalab_lam_cv,add:spline_interp}.{probe.mean,probe_gap_vs_add_chord_raw.mean,probe_n_better_than_chord_raw}`; `decomposition.L22_labels.headline_seed0.c_norm.chord_raw_at_spline_smooth_norm.value` (4.34); per-arc head-to-head counts (11/16, 10/16) and the headline residual (+2.71°) from `results/endpoint_diagnosis_raw/{L22_labels,L12_labels}.json` `[seed].arms.<arm>.probe_err`, seeds 1–16 (seed 0 is the headline arc). Path radius 0.86 / 0.63: `results/p2_steer_direction_direction_L12_contiguous_rawchord.json` `summary.{manifold,linear_raw}.overall.probe_radius_min`.
[^vs2]: `results/p5_velocity_sheet_v2.json` (mtime 17:36:53 ET; commit d97502d, dirty; CPU, 381 s) `results.{12,22}.{block2,block3,cross}.{joint,direction}.{own,matched}.{summary,gaps_sheet_minus}.<arm>.{err_dir,err_dir_mlp,err_spd,leak_spd,nearest_real_R}`, `results.<pt>.<design>.reader_floor.real.summary.real_target_clips.{err_dir,err_spd}`, `results.<pt>.block2.joint_path.own.summary.{sheet,chord_sheet}.{min_dir_radius,path_err_dir}`, `results.22.block2.dose_direction.x{0.5,…,2.0}.summary.<arm>.leak_spd`, `results.<pt>.extrapolate.extrapolate.v<speed>.summary.<arm>.err_spd` (targets 3.766 inside, 4.234 / 4.469 / 4.938 beyond the 4.0 m/s maximum, `scripts/run_velocity_sheet_v2.py:extrapolate`), arm definitions in `keys`; `results/p5_accel_sheet_v2.json` (mtime 17:36:29 ET) same paths for points 12 and 21; `results/p5_velocity_sheet_predictor.json` (box 4, commit d97502d dirty; `predictor.own.<arm>.{dir_err_deg,R_speed_probe,R_disp_speed}`, `paired.{own,sheet_norm}.sheet_minus_<arm>_dir_err`, `reference.{unedited,twin_forecast}.dir_err_deg`, `compute.gpu_seconds_forward`); v1 norms `results/p5_velocity_sheet.json` `layers.<pt>.steer.{direction,joint}.summary.<arm>.edit_norm`. `figures/fig_velocity_sheet_v2.png`.
[^mg]: `results/p5_motion_geometry.json` (mtime 17:40 ET; parts in `results/p5_motion_geometry_parts/`): `exp1_whitened_metric.points.{vjepa2,random}.<pt>.{pairs.<a|b>.overlap,union.participation_ratio,null_isotropic_2x2_overlap.p95}`; `exp2_interference_leakage.sets.{speed,acceleration}.{12,22}.{matched,natural}.<edit>.<readout>.mean`, `.leakage_vs_overlap.{spearman_decoder_overlap,spearman_subspace_overlap}`, `.residualised.{12,22}.{fourier_ring_2d,raw_chord}.<projection>.{speed_leak_probe_over_spread,speed_leak_mlp_over_spread,minus_none}`; `exp2b_chord_leak_mechanism.points.<pt>.{r2_leak_on_imbalances,pearson_leak_vs_true_speed_imbalance}`; `exp3_within_vs_between_and_fourier.fourier_dirset.{12,22}.{arms,differences}` and `.joint_speedset.{12,19,22}.arms.<arm>.{dir_probe_err_deg,speed_probe_err_mps}`; `exp4_coordinate_search_and_exp5_shortcuts.{registration,points.<pt>.per_coordinate.<coord>.{encoding_r2_transfer_to_dirset,encoding_r2_transfer_ci95,decode_cv.theta_mae_deg},winner_by_point,random_init,exp5.{position_residualised,pixels,start_position_only_quadratic},holm_note}`; `mlp_readout_calibration.points.<pt>.dirset_mlp_dir_mae_test_kept_deg`; figures `fig_motion_subspace_angles.png`, `fig_motion_leakage.png`, `fig_motion_coordinate_grid.png`.
[^coastf]: `results/p2_coast_faithful_L{12,22}.json` (mtime 17:43 ET; `unsteered_err.probe`, `frac_pairs_dtheta_gt_90`, `aperture.<space>.{mean_overlap,selected,in_band}`, `diagnostics.<space>.trace_C_steer`, headline `arms.{chord_raw,coast_linear_caa,full|contr|jaeger|a{0.1,0.5}|b0.3|unc,full|random_eigvec|jaeger|a{0.1,0.5}|b0.3|unc,full_jaegerR|pos|-|a0.1|b1|unc}.{probe_own,probe_chordnorm,norm_ratio_to_chord}`; 16 arcs `arcs.arms.<arm>.{probe_chordnorm.mean,gap_vs_chord_probe_chordnorm.n_arcs_ci_above_zero}` and `arcs_full.arms.full_jaegerR|pos|-|a0.1|b1|unc.{probe_own,probe_chordnorm,norm_ratio_to_chord,gap_vs_chord_probe_own}`, `arcs.mean_overlap_range`; `arm_key` defines the arm names); `scripts/run_coast_faithful.py`. The 10° rotation bound is arccos(2√(1−β)/(2−β)) at β = 0.3. Jaeger 2014 is not in `refs/`, so the singular-case AND formula is as checked by the rerun's author, not by me.
[^sacc]: `results/p5_speed_accel_angles.json` (mtime 17:43:53 ET; section C stamped 21:38 UTC): `D_accel_vs_speed.{vjepa2,random}/<pt>.{cos_speed_accel_line,cos_disattenuated,reliability_speed,transfer.speed_probe_on_accel_clips.{slope_mps_per_mps2,kinematic_mean_speed_slope},matched_mean_speed.auroc_accel_vs_const_matched_mean_speed}`; `A_rate_vs_displacement.<pt>.speed.{rate_vs_displacement.{rate_frac,growth_frac},cos_u_time_dir,random_dir_abs_cos_p95,steer.{constant_u.ds_intercept_m,step_specific_U_t.ds_slope_on_tau}}`; `B_ring_transfer.{12,22}.{global.{speed_gain,dir_change_deg},paired_nearest_real_R_diff.own_minus_90}`; `C_metric_dose_extrapolation.spacing.{speed,acceleration}/{12,22}.{pc1_all_values.log_minus_linear{,_ci95},cumulative_chord_real,cumulative_chord_shuffled_labels}`, `.twins.velocity/{12,22}.target_in_range.{twin_frac_energy_on_line,twin_probe_gain}`.
[^tokr]: `results/p5_token_patching_random.json` (mtime 17:36 ET; commit d97502d, dirty; `control`, `pair_types.{random,posmatch}.readers.{forecast,encoder_output}.points.{8,12,22}.{obj,bg}.binding_fraction.{mean,ci95}`, `verdict_inputs`, `readers.*.cv_r2`); trained values from `results/p5_token_patching.json` (same paths, `readers.*.cv_r2`); `scripts/run_token_patching.py`.
[^rep]: `results/p5_repair_attribution_L12.json` (`attribution_summary.{spline12,rawchord12,random12,spline22}.{sum_attn_contribution,sum_mlp_contribution,survival_after_block,survival_final_block24}`, `config.{target_deg,n_carriers,arms}`); commit 3c13095, dirty.
[^natdir]: `results/session2_direction_natural_norm.json` (mtime 17:44 ET; `table.{12,22}.{own,natural}.{spline,chord,null}.{err_to_target,R}`, `unedited.err_to_target`, `paired.12.natural.spline_minus_chord_err`, `reproduction_of_session2_own_norm`); 200 carriers × 4 targets, 639 GPU-s. Survival `survival.{12,22}.{own,natural}.<arm>.pt{16,24}`, dose `table.12.natural.*.applied_norm_median`, `paired.12.{spline,chord}_natural_minus_own_err`, `null_far_value_deg`; disk-token arm `results/session2_direction_natural_norm_disktokens.json` (`table.<arm>.{err_to_target,R}`, `obj_token_fraction`, `dose`, 358 GPU-s).

[^bgctrl]: `results/session2_direction_natural_norm_bgcontrol.json` (commit 98ca808, dirty; 426 GPU-s; `table.{disk12,bg_count,bg_energy,disk16,disk19}.{err_to_target,R,edit_energy_median,edited_tokens_median}`, `paired_vs_disk12.*_minus_disk12_err`, `references.{unedited_err,twin_ceiling,pooled_natural_chord_L12}`, `disk12_reproduction_maxabs_err_deg` 0.0). 200 carriers × 4 targets; 95% intervals are carrier-bootstrap (seed 0). `bg_count`: same token count as the disk set, random non-disk subset, the disk tokens' magnitudes permuted; `bg_energy`: all non-disk tokens at equal magnitude, total energy matched; `disk16`/`disk19`: the disk-token chord planned and applied at points 16 / 19 (labels angle).
[^timef]: `results/p5_time_manifold_controls.json` (mtime 17:38 ET; `subset.n_clips_with_random_features` 1,536; `{speed/random/timepool,speed/vjepa2/timepool_same_subset}.clock.22.spline.{slow,mid,fast}.slope`, `.fast_over_slow_slope`, `.geometry.22.heldout_var_explained.centroids`, `.steer.{12,22}.summary.spline.time_err`, `static_disk_control`); the 768-clip values are in `results/p5_time_manifold_controls_subset768.json`.
[^timep]: `results/p5_time_predictor.json` (mtime 17:43 ET; box 5 GPU; `design`, `references.{win+2,rev_frames}.{advance_fraction,along_px,ctx_time_readout_change_steps,frac_heading_flipped_gt90}`, `arms.{12,22}.{sp+2,ch+2,rev_sp,rand+2}.{advance_fraction,along_px,across_abs_px,forecast_speed_change_mps,frac_heading_flipped_gt90}`, `gaps.{12,22}.sp+2_minus_{null-2,rand+2,ch+2}_along_px`); encoder-side steering errors from `results/p5_time_manifold_controls.json` `.steer.*.summary.spline.time_err`.
[^att]: `results/p1a_attentive_{direction,speed,acceleration}.json` (mtimes 17:37 / 17:43 / 17:49 ET; `layers[].{point,attentive.{cv_r2_mean,cv_mae_mean,best_config,sweep},meanpool_ridge.{cv_mean,cv_mae_mean}}`, `caveat_time_averaged_tokens`, `status`).
[^s3a]: `results/session3_acceleration_predictor.json` (commit 6b28042, dirty; groups on boxes 1/3/4, `compute.group_provenance`; 1,893 GPU-s): `predictor.{12,21}.{own,chord_norm,natural_norm}.{spline,chord,linear_raw,null}.{direct_err_to_target,direct_R_acceleration,mean_speed_change_mps}`, `paired.<pt>.<cond>.spline_minus_chord_direct_err`, `twin_reference.{unedited_err_to_target,twin_forecast_err_to_target,twin_mean_speed_change_mps,physical_mean_speed_change_mps}`, `readouts.{direct.test_r2,displacement.test_r2,displacement.true_centroid_test_r2}`, `design.holdout`.
