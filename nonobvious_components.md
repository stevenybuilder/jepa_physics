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
  Say it on the slide; do not fix it silently. The paper's finding ("acceleration decodable early, without a velocity
  intermediate") is reproducible only in this weakened sense.
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
- **Both MAEs, always.** MAE-to-target falling while MAE-to-true rises is the evidence of a genuine shift (paper C.12).
  One without the other can be an artefact.
- **64 directions buy a new axis.** Report the result against angular shift |θ − θ\*|, not only vs N. Large shifts are
  where a linear edit is expected to struggle, which is the bridge to Part 2.
- **Norm drift per clip.** A steer that "works" at 10× the activation norm has left the data distribution. Log
  ‖x\*‖/‖x‖ per clip, not the mean.
- **The readout is at the same layer.** The paper's steering result is read at the steered layer. Whether later layers
  keep the edit is a separate claim (Part 2 needs it; §6 of spec.md, propagation test).

## Part 2: open-ended, where the non-obvious choices are the work

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
- **Direction is entangled with speed and motion type in the direction set.** A centroid per direction averages over
  speeds. Check whether the ring's radius depends on speed (a cone, not a circle); if it does, a single spline is the
  wrong object and the manifold is 2-D.

### 5. What constitutes a meaningful held-out steering evaluation

Three separations, each removing one way to fool ourselves:

1. **Held-out label values.** Every 4th value (interior for speed/acceleration) never used as a spline knot or in the
   steering subspace. Steering toward them tests the manifold, not the memory of 64 centroids. A spline through all 64
   centroids evaluated on those 64 is interpolation by construction. Precedent: Kantamneni & Tegmark 2025, App. C.2.
2. **A readout that did not build the intervention.** Evaluation probe on clips disjoint from both the knots' clips
   and the steered clips.
3. **Agreement with the model's own behaviour.** The trustworthy readout from jepa_steering: compare the steered
   activation with V-JEPA's own activation on a *real* clip that has the target value, at the steered layer and at
   every later layer (and, if run, through the predictor). Edits that matched at one layer diverged later in 42–59% of
   cases there. Goodfire's metrics at the steered layer cannot see this.

Plus: report **both directions of steering** (up and down the value range), **dose–response** along the path (K
waypoints, not just the endpoint), and **off-target effect** (steer speed, read direction; steer direction, read speed).

### 6. Controls that make spline-vs-line a result rather than a demo

- **Random-curve control.** A spline through shuffled-label centroids, or a random smooth curve of matched length.
  In jepa_steering, learned edits never beat matched random ones on the outcome; assume nothing.
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
later-layer readout is more convincing than a probe.
