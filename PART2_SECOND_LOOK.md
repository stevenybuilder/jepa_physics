# Part 2 second look: the Goodfire paper against what we did

Adversarial re-read of Goodfire (Wurgaft et al., arXiv 2605.05115; all sections and App. A–C) against Part 2. REPORT.md
and talk/ were not edited. New scripts and results were run from a frozen worktree at commit 33a2cbd (clean
provenance): `scripts/run_ring_occupancy.py` → `results/p2_ring_occupancy_L{12,22}.json`,
`scripts/run_isometry_linear.py` → `results/p2_isometry_linear.json`, `scripts/summarize_shift_dependence.py` →
`results/p2_shift_dependence.json`.

**Headline.** Three things change or sharpen the Part 2 story.

1. **The ring is a dense annulus in its own plane.** It is occupied along its whole length and its interior is empty.
2. **In the space the edit acts on, the chord does not leave the data more than the spline does.** In PCA-64 and in
   full activation space, chord waypoints are as close to real clips as spline waypoints at point 12, and closer at
   point 22 (17/17 arcs).
3. **A readout built from neither a probe nor the spline tells the two paths apart along the route.** The labels of
   the 10 nearest real clips (the kNN label readout) put the spline midpoint on the intermediate direction and the
   chord midpoint on a mixture of directions (17/17 arcs at both points).

Separately, the pre-registered growth of the path gap with angular shift holds in the stored runs but is not cited in
REPORT §4. Goodfire's isometry result reproduces with the predictor's own forecast as the behaviour space at points 8
and 12, and reverses at point 22.

---

## A. Claim-by-claim map

Status key: **C** covered, **P** partially covered, **M** missed, **NA** not applicable.

### A.1 Geometry claims

| # | Goodfire claim / method (section) | What we did (file, number) | Status |
|---|---|---|---|
| 1 | Concepts lie on a **nonlinear, curved manifold** in a low-dimensional subspace; linear steering assumes flat geometry (§1, §6 related work, §7) | Ring found at points 8/12/22 (`p2_geometry_direction_L*.json`). Centroid participation ratio (PR) 3.41–3.75, where a planar circle is 2, so the ring carries harmonics. REPORT §3.2: "rank-2 plus a second harmonic". | C |
| 2 | **Dense** manifold: activations "occupy a curved, low-dimensional manifold" (mountain car §5, B.1: 100 rollouts, 100 bins). Dense is used for the vertex set in A.5 and the bin grid in B.1. The LLM tasks have only 7–91 discrete values, and their "intermediate states" are decoded from the spline, never observed. | Previously only centroid-level. **New:** held-out clips in chart units (ring = 1) have 5th percentile radius 0.55 and 3.6% below 0.5 (point 12). The largest angular gap is 2.9°; every 5.6° bin holds 6–25 clips. Within-value SD is 2.2× the spacing of neighbouring values (4.4× at point 22), so the clouds form a continuum (`p2_ring_occupancy_L12.json` occupancy). | P → now C in the ring plane (see D) |
| 3 | **Isometry** M_h ↔ M_y: geodesic r = 0.89–0.999 vs linear r = 0.36–0.89 (§2.3, A.5); mountain car 0.996 vs 0.06 (C.3) | `run_part2` stores the geodesic r only: `isometry_probe` 0.83, `isometry_behaviour` 0.86 (point 12 contiguous). There is no chord baseline and REPORT does not cite them. **New** (`p2_isometry_linear.json`, smoothing spline, predictor-forecast M_y): point 8: geodesic 0.86 vs linear 0.80; point 12: 0.96 vs 0.90; point 22: 0.69 vs 0.78 (reversed). The encoder output, the same-layer Eq. 9 and the concept metric \|Δθ\| give the same numbers within 0.03. | P → now C (with caveats, B.2) |
| 4 | **Intermediate states**: manifold steering passes mass through adjacent concepts; linear "teleports" and puts mass on "other" at the midpoint (§3.2, Fig. 4) | Eq. 9 intermediate mass 0.68 vs 0.48, ordering 0.90 vs 0.79 (circular at the steered layer). Two-route test: MLP midpoint +89° / −94°, chord midpoint circular SD 140° (exploratory, spline on all 64 values). **New, held-out and non-probe:** at the midpoint, the kNN label readout is 16° closer to the nominal intermediate direction for the spline (−38° at shifts ≥ 120°). Its resultant length is +0.12 higher (+0.26 at large shifts). Both hold on 17/17 arcs at point 12; at point 22 the numbers are −19.6° and +0.10, also 17/17. | P → C |
| 5 | **Off-manifold regions / low density**: linear paths "cut through low-density regions" (Fig. 1, §1, §3.2); density geometry G_E ∝ (αe^{−E}+β)^{−1} (§3.4 Eq. 6) | G_E is never estimated in the paper; manifold steering is asserted to follow its geodesics. Ours: distance to curve, and 5-NN distance in full space (`excess_to_nearest_real`). **The stored runs show the spline farther from real clips than the chord**: point 22 +0.28 ± 0.11 over 17 runs (CI > 0 in 17/17); point 12 +0.17 (`p2_shift_dependence.json`). **New** density ratios (5-NN distance ÷ the real-clip median, `p2_ring_occupancy_*`): in the **2-D ring plane** the chord midpoint is farther at point 12 (spline − chord −0.14, −0.71 at shifts ≥ 120°; 14/17 arcs). In **PCA-64 and full space** there is no difference at point 12 (±0.01, CIs split). At point 22 the **spline is farther** (+0.037, 17/17; ring plane +0.15, 14/17), and off-support waypoints are +3.3 pp for the spline. | P, and it **contradicts** the premise outside the ring plane |
| 6 | Behaviour manifold M_y fit to real outputs in Hellinger space, with a tangent-plane spline (§2.2, A.4) | Implemented literally (`BehaviourManifold`, log/exp map). | C |
| 7 | World-model M_y = softmax of −‖z − μ_b‖/τ over activation-spline bins (Eq. 9/10, B.1) | Implemented as written, and flagged as circular at the steered layer. **Missed point for the talk:** Goodfire's own world-model isometry (r = 0.996) uses this same activation-defined F (Eq. 10: μ_b = γ_M(p_b)), so their world-model result shares our circularity. | C (the critique is not voiced) |
| 8 | How k is chosen: PCA-64 fixed, unjustified; pullback uses the top 32 (A.3, A.8) | k = 64 fixed; no sensitivity sweep. | P (M for the sweep; B.6) |
| 9 | Intrinsic coordinate for cyclic tasks: θ = atan2(PC2, PC1) "in the top two principal components of the activation subspace" (A.3) | The literal activation-PCA plane **fails** at points 8 and 12 (point 12: circular corr −0.58, order not preserved). The label-free angle comes from the top-2 PCs of the **centroids** (−0.983). At point 22 the activation plane works. REPORT §4.1 names the centroid plane, but "recovered without labels" should say "in the centroid plane". | C (deviation stated; wording) |
| 10 | Spline type and knots: interpolating natural or periodic cubic spline through every centroid, no smoothing (A.3); mountain car uses a √count-weighted smoothing spline (B.1); thin-plate spline with ghost points for the cylinder (A.3) | Periodic / natural cubic splines. Interpolating overshoots (held-out reconstruction 188 vs 2.8), so steering defaults to the B.1-style smoothing spline (weights √count/sd, s = m). TPS for the position sheet. No ghost-point cylinder. Downstream, the interpolating spline won the predictor test and the smoothing spline lost (REPORT §4.5). | C; tension noted in C.2 |
| 11 | Geodesics = cumulative Euclidean length along the spline in PCA-64, 150 sub-intervals; Hellinger on M_y (A.5) | Same (`arc_length`, `BehaviourManifold.geodesic`). | C |
| 12 | Isometry vertices: W centroids plus K interior points, with shared-geodesic pairs excluded; K = 0 for W ≥ 81 (A.5) | K = 0 with W = 64 (as for Goodfire's dense tasks). | C |
| 13 | MDS embeddings of geodesic vs linear distances (Figs. 2, 3, 6, 7b) | Not done. | M (cosmetic; the chord metric of a ring embeds as a ring) |
| 14 | Periodic / circular variables: periodic spline; cylinder with ghost points (A.3); mountain car closes because the wall and goal look alike (§5) | Periodic spline; transport arm for the rotating frame; cone check. No 2-D (θ, speed) TPS cylinder. | P |
| 15 | Uncertainty as a second manifold dimension: concentric circles by addition value with rising entropy (C.2, Fig. 11) | Not framed that way, but measured: ring radius grows with speed then saturates (velocity plane). **New:** speed-set clips at 0.25–0.67 m/s sit at chart radius 0.59 (38% below 0.5) with a chart-angle error of 30°, against 7° above 1.7 m/s. The ring interior holds slow, direction-uncertain clips, which is Goodfire's Fig. 11 phenomenon. | P (new link) |
| 16 | Factored control in 2-D: steering one coordinate leaves the other fixed (§4) | Off-target speed readout at the 180° chord midpoint (ratio 0.99). Rotating-speed-axis detector. Position-sheet TPS: no path advantage. | P |

### A.2 Steering method, evaluation and metrics

| # | Goodfire | Ours | Status |
|---|---|---|---|
| 17 | Linear = interpolation in ambient space (Eq. 1); manifold = interpolation in intrinsic coordinates (Eq. 2) | Same, with an additive (shift) form as the main arm. | C |
| 18 | **Linear baseline erases** the whole residual stream (replaced by the chord point between raw centroids). The manifold arm keeps the off-PCA-64 residual (A.6). Support therefore changes along with geometry. | Matched-support arms. Goodfire's pair run once: residual-erasing linear R 0.625 vs their manifold 0.547 (REPORT §4.4), so under their own recipe linear wins the endpoint on our data. | C (beyond the paper) |
| 19 | Base prompts: 16 fixed prompts with arbitrary answers, reused across pairs. Paths run centroid a → centroid b regardless of the prompt's own answer (A.6). | Carriers start at their own true value; 48 test clips per target. | C (deviation, better; not stated as a contrast) |
| 20 | K = 50 waypoints; up to 50 pairs (A.6) | K = 50; all shifts 0–180° to 8 targets. | C |
| 21 | **Held-out evaluation:** none. Centroids, splines and steering prompts all come from one task distribution; there are no held-out values; the pullback draws fresh prompts per pair (A.8). | Three held-out designs; knot, probe and test roles; 16 arcs; 70/30 rerun. | C (beyond) |
| 22 | **Controls:** none besides linear. No random curves, reflected or dose/norm matching, precision checks or shuffles. | 20 endpoint-matched random curves, unmatched random, shuffled, reflected, projected, dose-matched, BF16, norm-matched predictor. | C (beyond) |
| 23 | Extrapolation: nothing is said. Paths run only between fitted centroids. The mountain-car pullback degrades at the range end (the wall, tighter curvature, C.3). | Scalar extrapolation design: the spline is worse (speed point 12: 3.22 vs 0.158 m/s). | C (beyond) |
| 24 | Naturalness E_BC = Σ Bhattacharyya distance to M_y over waypoints, mean ± SE over pairs, paired t-tests; 2.8× gain (§3.2, A.7) | Implemented literally; `over_pairs` mean ± SE. Relative to the real-clip floor: 0.84 vs 1.42 (point 12), 1.01 vs 1.00 (point 22). Circular at the steered layer. | C |
| 25 | Shift dependence: not reported by Goodfire. Our pre-registered prediction: the gap grows with \|Δθ\| (PART2_RATIONALE §4). | Stored in every steering JSON and plotted (`fig4_gap_vs_shift_*`), **but not cited in REPORT §4**. **New summary over 17 runs:** the minimum readout radius gap is −0.003 (0–30°), +0.05, +0.12, +0.25, +0.43 and +0.62 (150–180°) at point 12, with CI > 0 in 17/17 runs from 90° on (16/17 at 60–90°). Point 22 is similar (+0.01 → +0.58). Endpoint error and nearest-real R show no trend. | P → C |
| 26 | Pullback (§3.3, A.8, A.9, C.3): optimise an activation path whose outputs follow the M_y geodesic, then score intrinsic R² against the manifold path vs the chord (0.47–0.78 vs 0.23–0.42) | Run as the angle-only "reverse test" at 21b27b6 (session2_reverse_path.json; one path per carrier, additive, Adam, norm cap); Goodfire-recipe rerun in progress (session2_pullback_goodfire.json). | **M** (GPU, B.1b) |
| 27 | Riemannian unification: G_I, G_E, G_F (§3.4, Def. 1) | G_E is approximated by the new kNN density ratio (row 5). G_F is not done. | P |
| 28 | Mountain car decoded frames at intermediate waypoints: blurred car on linear, a moving car on manifold (Fig. 7) | We have no decoder. Predictor forecasts were read at **endpoints only** (session 2), never along the path. | **M** (GPU, B.1a) |
| 29 | Probability spread / entropy is larger on the linear path (§5) | Eq. 9 entropy gap −0.27 (circular). kNN-label resultant length (new, non-circular) is lower for the chord. | C |
| 30 | Metrics used: isometry Pearson r, MDS, E_BC, pullback intrinsic R², mean distance to M_h, neighbourhood accuracy (ICLR), probability trajectories | All except MDS and neighbourhood accuracy (NA: no graph task); pullback R² pending in the Goodfire-recipe rerun. | P |
| 31 | Failure cases: pullback near the wall; months are the weakest isometry. Limitations: simple domains, token-level outputs, fitting needs ground-truth coordinates, intermediate variables untested. | Ours: point-22 endpoint on 3/16 arcs, scalar extrapolation, position sheet, washout at points ≤ 12, smoothing spline worst downstream. Our variables are intermediate (encoder, not output), which is the paper's own last limitation. | C (beyond) |

---

## B. Genuine gaps, ranked (impact on conclusions × feasibility before tomorrow evening Eastern)

### B.1 Predictor readout along the path (GPU, box 53030966, ~20–30 min)

This is the top gap and it was not run. Goodfire's central experiment is what happens at intermediate waypoints
(Figs. 4, 7). Ours reads the predictor only at endpoints.

- **Run:** at point 22 (and point 12 as a washout contrast), take 200 carriers × targets at shifts ≥ 120° plus exact
  180°. Use the chord, the interpolating spline and the smoothing spline, with 9 waypoints each, all norm-matched per
  waypoint. Run blocks 23–24 plus the predictor exactly as `session2_norm_matched.forward` does. Read the forecast
  with the predictor-native probes (angle and (sin, cos) radius). Also read kNN labels and density among the
  unedited forecasts of probe clips (`pred_pooled_all.npy`), and Eq. 9 E_BC on an M_y built from forecasts, which is
  non-circular.
- **What would change:** if the chord midpoint's forecast collapses (low radius, mixed kNN labels) while the spline's
  reads the intermediate direction, then REPORT §6 "rung 4 only by circular measures" and §4.3 "readouts that did not
  build the edit do not separate the two" become "the model's own forecast follows the ring on the spline and
  superposes on the chord". That is Goodfire's Fig. 7 claim with a real behaviour readout. If the forecasts do not
  separate, the geometric path result stays encoder-local, and the talk can say so with a direct test instead of an
  inference.
- **Cost:** ~5,400 forward passes at point 22 (session 2 ran ~19k in under an hour).

### B.1b Pullback through the predictor (GPU, ~45–60 min)

- **Run:** following A.8, a 10-control-point cubic path in PCA-64 at point 22, L-BFGS, 16 carriers. The target is
  the native-probe forecast angle along the arc at unit radius (the M_y geodesic). Score intrinsic R² against the
  spline and against the chord (A.9), on about 10 pairs including 180°.
- **What would change:** it adds the M_y → M_h direction. The bidirectional claim is Goodfire's main thesis, and we
  currently cover neither direction non-circularly along the path.

### B.2 Isometry with the chord baseline and non-circular behaviour spaces (CPU, ran)

`p2_isometry_linear.json`, smoothing spline.

| Point | Geodesic vs predictor forecast | Chord vs predictor forecast | Encoder out, geodesic / chord |
|---|---|---|---|
| 8 | 0.86 | 0.80 | 0.88 / 0.80 |
| 12 | 0.96 | 0.90 | 0.96 / 0.89 |
| 22 | 0.69 | 0.78 | 0.71 / 0.78 |

- With the interpolating spline the geodesic r drops at every point (point 12: 0.91; point 22: 0.58), because arc
  length accumulates centroid wiggle. Bootstrap intervals for the geodesic r are wide (point 12: [0.75, 0.98]) and
  sit below the point estimates, since resampling adds centroid noise.
- **Reading.** Goodfire's isometry reproduces at points 8 and 12, with a gap (+0.06–0.07) like their weekdays gap
  (0.99 vs 0.89), using the V-JEPA 2 predictor's own forecast as behaviour. At point 22 the chord is the better
  isometry.
- **Caveat worth a sentence in the talk:** every behaviour space gives the same r within 0.03 as the bare concept
  metric \|Δθ\|. So on a ring, this test measures "arc length along the activation spline is proportional to Δθ".
  It does not measure a representation–behaviour link. Two evenly spaced closed loops have geodesic r = 1 whatever
  their shapes.
- **Sentence to add (REPORT §4.1 or §4.3):** "Goodfire's isometry holds at points 8/12 against the predictor's own
  forecasts (0.96 vs 0.90 for the chord at point 12) and fails at point 22 (0.69 vs 0.78)."

### B.3 Density and occupancy along the path (CPU, ran; see D)

- **What changes:** REPORT §4.3's "the line cuts through it" is true in the ring plane at point 12. It is false as a
  statement about density in the space the edit acts on. At point 22 the spline is the less natural path by every
  real-clip density measure (17/17 arcs). REPORT and the talk should say "cuts through the ring's plane" and cite
  the density null, or they overclaim Goodfire's "low-density region" premise.

### B.4 Shift dependence not cited (no compute, ran)

- `p2_shift_dependence.json` confirms the pre-registered prediction on 17 runs (numbers in A.2 row 25).
- **Sentence to add:** "The path gap is zero below 30° and grows monotonically to +0.62 at 150–180° (17/17 runs from
  60°); endpoint error shows no trend with shift."

### B.5 Two-route test with held-out values (CPU, ~15 min; not run)

- The two-route 180° result is labelled exploratory because the spline is built on all 64 values.
- **Run:** repeat it with the antipode inside a held-out contiguous arc, 16 seeds.
- **What would change:** it would promote the cleanest intermediate-state result from exploratory to held-out.

### B.6 k sensitivity (CPU, ~15 min; not run)

- **Run:** k ∈ {8, 16, 32, 128} for the geometry checks and the contiguous path metrics.
- **Impact:** low. The ring lies in the top centroid PCs, so this pre-empts a reviewer question and is unlikely to
  change anything.

### B.7 Joint density test for the cone (CPU, ~15 min; not run)

- The chord's 180° midpoint lands in the chart region where slow speed-set clips live (radius < 0.5), but an MLP
  still reads it as fast (speed ratio 0.99).
- **Run:** measure 5-NN density in (chart plane × speed axis) against a pooled direction-plus-speed reference. This
  would show whether the midpoint is an unrealised combination: fast speed with no direction.
- **What would change:** the "off-manifold" claim could be stated jointly rather than only in the plane.

### B.8 Radial (uncertainty) steering (GPU, ~15 min; not run)

- **Run:** shrink the radius at a fixed angle (Goodfire Fig. 11 analogue) and read the forecast's angle spread and
  speed.
- **Impact:** a new, positive handle, but it does not change the current conclusions. Low priority.

---

## C. Beyond the paper, and silent deviations

### C.1 Where Part 2 goes beyond Goodfire (safe to say in the talk)

- **Held-out label values.** Three designs, 16 arcs, disjoint knot, probe and test roles. Goodfire has no held-out
  values and draws its steering prompts from the same distribution as its centroids.
- **Controls Goodfire lacks.** Matched support, random endpoint-matched curves, a reflected bend (which tests bending
  the right way), a projected arm, dose and norm matching, and BF16.
- **Their own baseline, on our data.** Goodfire's residual-erasing linear arm beats their manifold arm on endpoint
  agreement.
- **Downstream readouts.** Propagation through later layers and the predictor. Goodfire reads only the same model's
  output head. Our variables are intermediate, which is their stated open limitation.
- **The circularity of the Eq. 9 world-model M_y is named.** Goodfire uses the same construction for their r = 0.996
  without comment.
- **New in this pass:** a real-clip density test of the "low-density" premise (Goodfire defines G_E but never
  estimates it), a non-probe kNN readout of intermediate states, and isometry against the predictor's forecast.

### C.2 Deviations from Goodfire's recipe that REPORT does not state, or understates

- **Label-free angle.** It is taken in the centroid PC plane; Goodfire's literal activation plane fails at points 8
  and 12. The REPORT §4.1 table shows this, but the summary's "recovered without labels" omits it.
- **Smoothing spline as default.** Goodfire's language tasks use an interpolating spline. The smoothing spline
  follows their B.1 but with our own s = m. The default was chosen by centroid reconstruction on train, and it is the
  worst arm downstream. REPORT states the downstream fact but not that the choice departs from A.3.
- **Carriers start at their own value.** Goodfire starts from centroid a on prompts with arbitrary answers. This is
  better, but it is a different estimand and is not contrasted.
- **Eq. 9 scale.** τ = 0.5 is applied literally in PCA-64 units of ViT-L, whereas Goodfire's units are LayerNorm'd
  64-d encoder outputs. A τ sweep is stored (`tau_sensitivity`), but REPORT does not mention the scale mismatch.
- **Stored isometry uses kept values only.** `run_part2` computes it on 56 knots with no chord baseline, and REPORT
  never reports it.
- **Additive shift arm.** It moves each clip's offset rigidly. At large shifts this places outward offsets on the
  inner side, which is why the transport arm exists. The density results suggest the rigid shift is part of why the
  spline is farther from real clips at point 22, though transport does not fix it (+0.059 PCA-64, 17/17).

---

## D. Is direction a dense, curved manifold in Goodfire's sense?

Short answer: it is a dense, occupied annulus in the 2-D ring plane. It is not a detectable low-density hole in the
64-D or full space the steering acts on. The previous results showed only a ring of centroids.

### What the earlier results established

- A label-free ring of centroids: circular corr −0.983 at point 12 (`p2_geometry_direction_L12.json`).
- Curvature below centroid noise over gaps of 45° or less (sagitta ÷ noise 0.22), detectable at 90°. The
  leave-one-out cubic wins only at stride 16. The planted ring is recovered once its radius is at least 0.4 of the
  real ring's.
- Centroid PR 3.41 against residual PR 7.84. The clip cloud is a thick tube around the ring (≈8 nuisance
  dimensions: speed, position, motion type).
- Velocity plane: ring over velocity plane, with the radius saturating in speed. Cosine tangent: the spline tangent
  bends away from the chord (0.31 / 0.41 at the ends).
- All of this is centroid geometry. The only clip-level number, the planted-ring `real_ring.r_within_sd` = 4.8
  (radius ÷ within-value SD in the chart plane), implied a hole but was never reported as occupancy. The steering
  "off-manifold" metrics were distance to a spline (circular), Eq. 9 (circular), or full-space 5-NN, where the
  spline was already worse than the chord.

### What the new occupancy test shows (`p2_ring_occupancy_L12.json`, held-out probe and test clips, chart units, ring = 1)

- **Occupied along its whole length.** The largest angular gap between clips is 2.9°, and every 5.6° bin has 6–25
  clips.
- **A continuum.** Within-value SD is 0.21, against a neighbour spacing of 0.098 (overlap ratio 2.2; 4.4 at point 22).
- **An empty interior** on the direction set. The 1st, 5th and 50th percentile radii are 0.36, 0.55 and 1.04; 0.26%
  of clips lie below 0.25. The chord midpoint between centroids 180° apart sits at radius 0.09, where 0% of held-out
  clips lie; at 135° it sits at 0.39 (1.8% of clips inside). Point 22 is the same (0.14; 0.5%).
- **The cone is real.** The interior is populated only by the slowest speed-set clips (0.25–0.67 m/s: median radius
  0.59, 38% below 0.5), whose direction is uncertain (chart-angle error 30°).
- **Intermediate states are realised by real clips of the intermediate direction.** At the spline midpoint, the 10
  nearest real clips (PCA-64) read the nominal intermediate direction to within 21° at 150–180° shifts, with
  resultant length 0.92. At the chord midpoint they read 78° off, with resultant 0.66 (a mixture). This holds on
  17/17 arcs at points 12 and 22.

### What is missing, and what cuts against the claim

- **The hole is a 2-D phenomenon.** In PCA-64 the chord midpoint's 5-NN distance is 1.16× the real-clip median,
  against 1.13× for the spline (150–180°, point 12). Across 17 arcs the gap is +0.007 (CIs split 4 above / 9 below).
- **At point 22 the spline waypoints are the farther ones** in every space (PCA-64 +0.037, ring plane +0.15, 17/17
  and 14/17 arcs).
- So Goodfire's "linear steering passes through low-density regions" holds for the ring's plane. It is not
  measurable in the representation as a whole, where the ring is 2 of roughly 10 structured dimensions around each
  point.

### To make the full claim

- **Show a density hole in the space the model reads, not only in the chart plane.** Two routes: density in a
  whitened or LEACE-projected direction subspace (CPU, 15 min), or, decisively, whether the model's downstream
  computation treats the chord midpoint as unnatural (B.1: forecast radius, kNN labels and density among real
  forecasts along the path).
- **Test the manifold's dimension explicitly.** Run a local-PCA intrinsic-dimension estimate on held-out clips per
  angle bin, to say "a 1-D ring times an ≈8-D nuisance tube" rather than "a curve" (CPU, 10 min, not run).

---

## Top five gaps

1. **Predictor readout along the path, chord vs spline at point 22.** GPU, ~20–30 min. Not run; this is the one
   test that could turn the path result from circular into behavioural.
2. **Pullback through the predictor (Goodfire's M_y → M_h direction).** GPU, ~45–60 min. Not run.
3. **Real-clip density along paths.** CPU, ran. The chord crosses an empty interior only in the 2-D ring plane. In
   PCA-64 and full space it is no farther from real clips than the spline at point 12, and closer at point 22
   (17/17 arcs). REPORT's "cuts through" needs scoping.
4. **Isometry with the chord baseline and the predictor's forecast as behaviour.** CPU, ran. Geodesic 0.96 vs chord
   0.90 at point 12 and 0.86 vs 0.80 at point 8; reversed at point 22 (0.69 vs 0.78). On a ring the metric reduces
   to arc length ∝ Δθ.
5. **Shift dependence is uncited.** No compute, ran. The radius gap grows from 0 below 30° to +0.62 at 150–180°
   (17/17 runs from 90°, 16/17 at 60–90°); endpoint error and nearest-real R stay flat.
