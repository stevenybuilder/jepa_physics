# Non-obvious components

The README fixes what to build. This file lists the places where the obvious implementation is wrong, thin, or
misses something the reviewer will care about. Part 1 is a reproduction, so its non-obvious parts are judgment calls
inside a fixed method. Part 2 is open-ended, so its non-obvious parts are design choices that separate a demo from a
finding. Sources: `physics paper.pdf`, `steering paper.pdf`, `JEPA_STEERING_LESSONS.md`, `MEMJEPA_LESSONS.md`.

## Part 1: reproduction, with judgment

Faithful to the paper's method (App. B, C.11, C.12). Not blind: each step has a reading that needs a rule.

### 1. Layer-wise probing: "identify where each variable becomes available"

- **Availability needs a rule, not an eye.** State it before plotting: e.g. the first layer whose CV score reaches
  90% of that variable's maximum, with a bootstrap CI on the layer index. Report three things per variable, as the
  paper's finding has three parts: onset, peak, and the decline toward the output.
- **The x-axis is layer fraction**, as in the paper, so "one third depth" is checkable.
- **Direction's expected curve is gradual under mean pooling.** The paper's own App. C.5 / Fig. 18a says mean-pooled
  direction rises gradually while per-patch direction jumps at the emergence zone. A gradual curve is the expected
  reproduction. The disk-token pool (mean over the patches the disk covers) is the cheap check of the sharp version.
- **Direction is circular in every line of code.** Targets (sin θ, cos θ); decode with atan2; error is wrap-aware
  circular MAE in degrees; never average raw degrees; R² on sin/cos can look fine while angle error is bad, so report both.
- **Acceleration is confounded in this data.** Every acceleration clip starts at rest, so acceleration, mean speed
  and total displacement are one variable. An "acceleration probe" cannot be separated from a mean-speed probe here.
  Say it on the slide; do not fix it silently. The paper's own acceleration set has the same confound (App. A.1.2,
  sphere initialised at rest), so its "acceleration decodable without a velocity intermediate" claim is untested by
  its data and by ours. That is a sharper slide than "our data is weaker".
- **Is direction Cartesian or polar?** The paper asserts polar factorisation dominates (Table 1) but only plots vx and
  ax. Our 64 × 64 design tests it directly: onset layer of (vx, vy) vs (sin θ, cos θ) on constant-speed clips, same
  availability rule. If Cartesian comes up first, direction's "emergence" may be the normalisation v/‖v‖, and a
  nonlinear-probe gain on θ may be nothing but atan2 of a linear (vx, vy) readout (jepa_steering's
  coordinate-transform null).
- **The direction set mixes motion types** (half constant speed, half accelerating). Report direction per motion type
  as well as pooled; a probe that works on one and not the other is a finding, not noise.
- **Two comparisons keep decodability honest** (cheap, from the lessons files, not from the paper): a shuffled-label
  probe and a pixel baseline. A disk on black is nearly pixel-decodable; claim "V-JEPA encodes X at layer L" only where
  the curve beats both. These are additions, kept out of the core figure if they clutter it.
- **Hook point and the last layer.** In the installed transformers, `hidden_states[24]` already has the final
  LayerNorm applied; the raw block-24 output needs a hook. Plot the post-LN point separately or it looks like a cliff.

### 2. Iterative nullspace probing: "dimensionality and redundancy"

- **Two readings from one curve.** Dimensionality is K at the paper's stopping threshold (2K for direction, K for
  scalars). Redundancy is the *shape* of the decay: a plateau followed by a cliff means many near-equivalent readouts
  (redundant code); a steady decline means a few strong directions and a tail. Report both, per variable.
- **The sawtooth is the qualitative finding to check**, not assume: the paper sees it for direction (paired sin/cos
  features) and not for speed. With 64 directions the pairing can be checked directly (angle between consecutive probe
  weights in the θ-response plane).
- **"Tens of dimensions" means nothing without a random-removal band.** Project out random subspaces of matched rank
  (10 seeds) and plot that curve behind the INLP curve. Redundancy is claimed only where INLP survives far longer
  than random removal predicts.
- **Regularisation fixes the count.** α is fixed at the layer-probe value for all rounds; per-round tuning inflates K.
  n ≈ 1,200 train clips vs d = 1024, so state that K is a CV-scored ridge count.
- **INLP removes linear information only.** After the last round the variable may still be nonlinearly present. Say
  "linearly removed", not "removed". An MLP probe on the residual is a one-line check if the claim matters.
- **One coordinate system.** Probes, projections and the steering basis all live in train-standardised space;
  nullspaces there are not orthogonal in raw space. Never mix.

### 3. Multi-probe subspace steering: "held-out data not used to construct the subspace"

- **Held-out cuts both ways.** The steering subspace never saw the steered clips, *and* the evaluation probe never saw
  the subspace's clips. The paper's protocol fits the evaluation probe on the test activations it then steers; that is
  what we reproduce, and it is stated as the paper's choice.
- **Never score steering with the probe that built the subspace.** The MemJEPA shared-scorer bug. A linear evaluation
  probe whose weights lie in span(V) reads back whatever was written there.
- **The evaluation probe's own R² must be out-of-fold.** The paper's 0.99 is in-sample from 103 clips in d = 1024.
  Choose its α by CV inside `test` and report the out-of-fold score; ~300 clips is still n < d.
- **Steer to a radius-matched target as well as the paper's unit vector.** Ridge readouts have radius < 1, so
  least squares toward (sin θ\*, cos θ\*) pushes activations beyond the data. Report the readout radius per clip
  beside the angle; atan2 of a near-zero readout is noise.
- **Both MAEs, always.** MAE-to-target falling while MAE-to-true rises is the evidence of a genuine shift (paper C.12).
  One without the other can be an artefact.
- **64 directions buy a new axis.** Report the result against angular shift |θ − θ\*|, not only vs N. Large shifts are
  where a linear edit is expected to struggle, which is the bridge to Part 2.
- **Norm drift per clip.** A steer that "works" at 10× the activation norm has left the data distribution. Log
  ‖x\*‖/‖x‖ per clip, not the mean.
- **The readout is at the same layer.** The paper's steering result is read at the steered layer. Whether later layers
  keep the edit is a separate claim (Part 2 needs it; §6 of spec.md, propagation test).

## Part 2: open-ended, where the non-obvious choices are the work

The high-level reasoning (why steering is the test of "used" rather than "readable", what spline steering does,
why direction's circular structure is the case that separates spline from line, and what jepa_steering's
nonlinearity results do and do not transfer) is in `PART2_RATIONALE.md`. Items §4–10 below are the checklist.

### 4. The circular structure of direction

- **The spline must close.** Periodic cubic spline; a natural spline through 64 points on a ring has a seam at 0°/360°
  that will show up as a steering failure at exactly one place.
- **Find the angle without labels.** Goodfire's intrinsic coordinate is atan2(PC2, PC1) on the centroids. Compute it
  unsupervised, then check it against the true θ. If they agree, the ring is in the top two PCs, which is itself a
  result (and matches the paper's ring-shaped population code). If they do not, the ring lives in more dimensions,
  which is also a result, and the intrinsic coordinate has to come from somewhere else (e.g. a 2-D embedding of the
  centroid loop, or the labels, stated honestly).
- **Linear steering through the centre is the predicted failure.** A straight path between opposite directions passes
  through the ring's centre, where the representation means "no direction". Report the spline-vs-line gap against
  angular shift; expect nothing at small shifts and a clear gap near 180°. The lessons file says: a spline beats a
  line only for large steers. If that is what we see, say so.
- **Endpoints coincide by construction.** Both paths end at the target centroid, so any readout at the endpoint of
  the steered layer cannot separate spline from line (confirmed on the synthetic ring: identical probe error and
  nearest-real agreement at every shift). The difference lives only (a) along the path, at intermediate waypoints
  (off-manifold energy, waypoint readouts), and (b) downstream, after the edit is run through later layers. So the
  Part 2 figures are dose–response along the path and propagation, not endpoint MAE.
- **Direction is entangled with speed and motion type in the direction set.** A centroid per direction averages over
  speeds. Check whether the ring's radius depends on speed (a cone, not a circle); if it does, a single spline is the
  wrong object and the manifold is 2-D.
- **Cone or cylinder, not ring.** The speed set crosses 64 speeds with 64 directions (24 clips per speed). If radius
  or centre moves with speed, the right object is Goodfire's cylinder task: a thin-plate spline over (θ, speed) with
  ghost points one period above and below θ (steering paper A.3, and its concentric-circles result). That is a
  stronger Part 2 than three separate 1-D splines, and it is the natural place to test off-target effects.
- **Speed may also be curved.** The paper's speed GLM is quadratic because neurons have preferred speeds (C.8). A
  population of peaked tuning curves traces a curved centroid path. Do not predict "speed is straight" before the
  knot-subsampling test.
- **The global mean of a ring is its centre.** A mean-difference or single-vector "direction" steer averages to
  ≈ 0. Steering must be target-conditioned (the Part 1 least squares per θ\*, or the spline). jepa_steering built a
  rotating-code detector for this reason and a circular chart X ≈ μ + A[cos θ, sin θ] with the phase decoded by atan2;
  use that chart as the supervised check on Goodfire's unsupervised atan2(PC2, PC1).

### 5. What constitutes a meaningful held-out steering evaluation

Three separations, each removing one way to fool ourselves:

1. **Held-out label values, three designs.** (a) Scattered: every 4th value held out. This is a local interpolation
   check only: the largest remaining gap is 11.25°, where a chord and a cubic nearly agree. (b) Contiguous: a 45°
   arc of direction (8 values) and an interior block of 8 speeds and 8 accelerations. The spline-vs-line claim rests
   on this one. (c) Extrapolation: the top 8 speeds and accelerations, labelled as such. A held-out value is also the
   one place an endpoint readout can separate spline from line: there is no centroid to land on, the line aims at a
   chord point and the spline at the curve point, and they differ by the sagitta. Plus held-out *context*: steer
   speed-set clips (speeds 0.25–4 m/s, starts in [−1.2, 1.2]²) with a direction spline built on the direction set
   (1–7 m/s, [−2, 2]²). A spline through all 64 centroids evaluated on those 64 is interpolation by construction.
   Precedent: Kantamneni & Tegmark 2025, App. C.2.
2. **A readout that did not build the intervention.** Evaluation probe on clips disjoint from both the knots' clips
   and the steered clips.
3. **Agreement with the model's own behaviour.** The trustworthy readout from jepa_steering: compare the steered
   activation with V-JEPA's own activation on a *real* clip that has the target value, at the steered layer and at
   every later layer (and, if run, through the predictor). Edits that matched at one layer diverged later in 42–59% of
   cases there. Goodfire's metrics at the steered layer cannot see this.

4. **Three senses of "held-out", named every time.** Excluded from fitting; excluded from development decisions
   (layer, K, knot count, spline type, α); untouched until the final read. jepa_steering conflated them and its
   +6.25 pp development gain was 0.00 pp on fresh scenarios. All Part 2 design choices are made on train folds;
   `test` is read once for the final table; anything chosen after that read is labelled exploratory.

Plus: report **both directions of steering** (up and down the value range), **dose–response** along the path (K
waypoints, not just the endpoint), and **off-target effect** (steer speed, read direction; steer direction, read speed).

### 6. Controls that make spline-vs-line a result rather than a demo

- **Random-curve control, ≥ 20 draws.** A spline through shuffled-label centroids, or a random smooth curve of
  matched length; separately, the learned coefficients applied through a random frame (orientation-only null). In
  jepa_steering, learned edits beat matched random only offline, never on behaviour; report the empirical rank, and
  report per-clip rescue and harm counts next to the net effect (its rank-4 edit rescued 23 failures and broke 17
  successes; a method can win most families and lose on average).
- **Reflected and projected arms.** Same endpoints, dose and waypoint count: the spline's bend flipped
  (2·chord − spline) and the chord traversed with the spline's spacing. A random curve tests bending at all; the
  reflected curve tests bending the right way. From jepa_steering's registered action-geometry protocol.
- **Matched support.** Spline and line edit the same subspace and keep the same off-subspace residual. Goodfire's
  linear baseline replaces the whole activation while its manifold arm keeps the residual, so its comparison mixes
  "residual kept vs erased" with "curved vs straight". Run Goodfire's version once, labelled.
- **The subspace-patching illusion.** An edit can move the readout through a pathway the model does not use for the
  variable (Makelov, Lange & Nanda 2023, arXiv 2311.17030). "MAE-to-true rises" does not rule this out; the
  later-layer agreement readout (§5.3) is the check.
- **Shared-vs-clip-specific Δ.** Split each steering edit into the part shared across clips and the clip-specific
  remainder. At ~99% shared (what jepa_steering found), a spline edit and a linear edit are nearly the same constant
  shift, and any difference between them is small by construction.
- **Precision.** Repeat the spline-vs-line comparison on BF16-rounded activations. In jepa_steering, rounding
  reversed which fit won. If the result survives, it is real; if not, that is the "one clean negative" for the talk.
- **Off-manifold energy of the Part 1 linear steer**, measured first. If the multi-probe edit already stays near the
  centroid curve, the spline has little room to win, and that is worth knowing before fitting one.

### 7. Cheap geometry checks before any spline (all on stored activations, minutes each)

- Leave-one-value-out: reconstruct each centroid from its neighbours with a line vs a cubic. This is the honest
  curvature test; jepa_steering's four-symmetric-point test could not see third-order structure.
- Within-value spread vs between-centroid curvature: if the noise cloud is larger than the bend, the spline is
  fitting noise.
- Participation ratio of the centroids vs of the within-value residuals.
- Principal angles between the Part 1 INLP basis and the centroid PCA plane: does the paper's "many probe directions"
  subspace contain the ring?
- Planted-ring positive control: plant a synthetic ring of known radius in real activations and check that the
  pipeline recovers it (intrinsic angle, knot-subsampling cubic gain) before reading any "no ring" or "no curvature"
  result. jepa_steering's planted rotating-code test is what let it report its null as a null.
- Direction-vs-speed principal angles from the INLP bases against the random expectation k_A/d (paper C.4). The
  paper never measures this pair; it comes before any off-target claim.
- Knot spacing along the speed curve: log-like or linear? A straight, evenly spaced speed curve means "the spline
  gives no advantage for speed", which is a finding.

### 8. Behaviour manifold: V-JEPA has no output distribution

Goodfire fits the behaviour manifold on output probabilities. V-JEPA has none. The stand-ins, in order of strength:
evaluation-probe readouts on real clips (weakest, a probe again); nearest real-clip agreement across later layers
(the readout in §5.3); the predictor's forecast of the future tokens compared with real future tokens of clips at
the target value (strongest, and the only one that is "behaviour"). Choose and state which one plays M_y; do not call
a probe readout a behaviour.

### 9. Things the lessons files say not to do

Attention-distance maps, search-entropy measures, factorial interaction studies, learned-operator searches,
symmetric four-point curvature tests, tiny reused evaluation sets, more benchmarks before the core question is
answered, six-plus analyses with no story. One sharp comparison with controls beats three shallow ones.

### 10. Framing for the talk

One question: are V-JEPA's physical variables straight directions or curved manifolds, and does steering along the
curve do anything a straight edit does not, as judged by the model's own later layers? Three rungs of evidence
(decodable → steerable at the layer → propagated). One clean negative. Backup slides for: why mean-pool, why that
layer, what "40 dimensions" means given INLP is linear-only, how the held-out evaluation avoids leaking, why the
later-layer readout is more convincing than a probe. Bar to hold ourselves to (jepa_steering's synthesis): a
semantic axis must predict a physical variable on held-out examples and interventions must change the targeted
variable while preserving the others; a manifold claim additionally needs a fitted chart or metric and an
on-manifold vs matched-linear comparison.
