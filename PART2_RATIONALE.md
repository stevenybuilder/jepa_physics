# Part 2 rationale: why steering, what spline steering is, and how to reason about it here

Companion to `spec.md` §6 and `nonobvious_components.md` §4–10. This file is the *why*. Those files are the *what*.
Sources: README, `physics paper.pdf` (Joseph et al. 2026), `steering paper.pdf` (Goodfire, Manifold Steering),
`JEPA_STEERING_LESSONS.md`, the local `jepa_steering` folder (`findings.md`, `analysis/out/fresh-v2/`).

## 1. The one-sentence framing

The take-home asks how V-JEPA *represents* direction, speed and acceleration. Probing answers "is the variable
readable". Steering answers "is the readable thing the thing the model actually uses". Part 2 is the second question
asked with a better geometric model of the representation than a straight line.

## 2. Why steering at all

A probe can succeed for three different reasons, and only one of them is interesting:

1. the model computes and uses the variable (what we want to claim),
2. the variable is trivially present in the input and the probe reads it off (a disk on black is nearly
   pixel-decodable; a random-init encoder will decode some of it),
3. the probe is picking up a correlate (acceleration ≡ mean speed ≡ displacement in this data).

Steering is the intervention that separates (1) from (2) and (3). If we write a new value of the variable into the
activation, using only the geometry we claim to understand, and the model's *later* computation behaves as if the
clip had that value, then the geometry is causally used. If the edit is read back only by the probe that defined it,
we have learned nothing about the model. This was the central lesson of jepa_steering: **readable is not usable**.
There, curvature was measurable to high precision, edits moved probe readouts and beat matched random edits by
1–1.5% of offline forecast error, and none of it changed task behaviour: the development gain was zero on fresh
scenarios.

So the honest hierarchy of evidence, which the talk should walk up rung by rung:

| Rung | Claim | Test | Part |
|---|---|---|---|
| 1 | The variable is linearly decodable at layer L | layer-wise probes, controls | 1 |
| 2 | It occupies a K-dim linear subspace, with redundancy | INLP curve vs random-removal band | 1 |
| 3 | Writing into that subspace changes the readout at L | multi-probe steering, held-out clips | 1 |
| 4 | The representation is curved, and a curved edit stays on-distribution where a straight one does not | spline vs line, off-manifold energy | 2 |
| 5 | Later layers / the predictor agree with the edit | propagation test against real target-value clips | 2 |

Part 1 ends at rung 3 because that is where the paper ends. Part 2 is rungs 4 and 5. Rung 5 is the only one that
counts as behaviour for an encoder with no output distribution.

## 3. What spline steering is, in plain terms

Linear steering (Part 1, and most of the literature) assumes that the variable is encoded along a fixed direction:
to make the model "see" a faster disk, add a multiple of the speed direction. That is a straight-line model of the
representation. Goodfire's observation is that when you plot the class-conditional mean activation (centroid) for each
value of a variable, the centroids do not lie on a line. For ordered scalars (ages, letters) they trace a smooth
curve; for cyclic variables (weekdays, months) they trace a closed loop. A straight edit from value *a* to value *b*
therefore cuts across the curve through regions the model never produces. Spline steering replaces the straight edit
with a path along the fitted curve:

1. reduce the activations to a low-dimensional subspace (PCA-64) where the curve is visible,
2. compute one centroid per value, fit a cubic spline through them (periodic for cyclic variables) and give the
   spline an intrinsic coordinate (arc length, or the angle atan2(PC2, PC1) for a loop),
3. to steer a clip from *a* to *b*, move its on-manifold component along the spline from the point for *a* to the
   point for *b*, and carry its orthogonal-complement component along unchanged,
4. measure how far the edit strayed from the data distribution (off-manifold energy) and whether the output changed
   the way the curve predicts (isometry between the activation curve and the output curve).

Goodfire's claim is that (3) beats a straight edit in exactly the places where the curve bends most, and that
distances along the activation curve predict distances along the behaviour curve. Nothing in that method is specific
to language models. It needs an activation, a labelled variable, and a behaviour to read. V-JEPA supplies the first
two directly and the third only via stand-ins (§7).

## 4. Why direction is the variable that makes Part 2 worth doing

Direction is cyclic. That single fact does most of the work in Part 2:

- **The centroids must form a closed loop, or the linear story is already wrong.** The physics paper reports a
  ring-shaped population code for direction and a sawtooth INLP curve because sin θ and cos θ are removed in pairs.
  A ring has no "direction direction". A single linear probe direction only reads one projection of the ring.
- **A straight edit between opposite directions passes through the centre of the ring.** The centre is where the
  representation means "no direction" (the mean over all directions). For small angular changes a straight chord and
  the arc are nearly the same, so linear steering should look fine. Near 180° they diverge maximally. This gives a
  parametric prediction: spline advantage grows with |θ − θ\*|, is ~0 below ~45°, and is largest at 180°. That
  prediction is falsifiable and is the cleanest result Part 2 can produce.
- **The spline has to be periodic.** A natural spline through 64 points on a loop has a seam at 0°/360°. A seam is a
  steering failure at exactly one place and a giveaway that the geometry was modelled wrongly.
- **The intrinsic coordinate should be found without labels.** If atan2(PC2, PC1) of the centroids recovers θ, the
  ring lies in the top two principal components and the model's coordinate matches the physical one. If it does not,
  the ring lives in more dimensions (a great circle tilted through several PCs, or a higher-frequency Fourier code);
  that is still a result, and the intrinsic coordinate must then come from a 2-D embedding of the loop, stated as such.
- **Direction may be a cone, not a circle.** The direction set mixes speeds and motion types. If the ring's radius
  grows with speed, direction and speed share a 2-D manifold and a 1-D spline is the wrong object. Check the radius
  vs speed before fitting anything.

Speed and acceleration are the contrast cases. Their centroids should trace an open curve. If that curve is nearly
straight with evenly spaced knots, spline and line coincide and the honest report is "no advantage for speed, and here
is why". If the knots compress at high speed (a log-like spacing), the curve is straight but the *parameterisation*
is nonlinear, which is a different and smaller finding: linear steering with a nonlinear dose schedule.

## 5. What jepa_steering already taught about nonlinearity, and what it did not

Corrected 27 Sep from the full audit of the local folder (`JEPA_STEERING_LESSONS.md` §5–§6 has the per-investigation
record with file paths; this is the summary that feeds Part 2).

**What it found (applies here):**

- **Curvature is real and measurable, and sits just above the arithmetic noise floor.** Cubic reconstruction of an
  omitted activation from its neighbours beat linear by 3–4 orders of magnitude in FP32 (Wall 3,967×, PointMaze
  7,020×); in BF16 cubic was 42–47% *worse*. A later controlled run showed rounding explains most but not all of the
  flip. Reading: the latent space is a dense, smoothly curved manifold, and at small offsets the bend is a
  second-order response only visible above fp32 noise. Any spline-vs-line gain here must beat within-value clip
  scatter and BF16 rounding of the pooled activations, and be compared at matched delivered dose (dose
  normalisation once turned a 99.9% fidelity win into a 16% loss).
- **Curvature did not select useful edits.** Four registered curvature arms (linear, cubic, projected-onto-chord,
  reflected-bend) were inconclusive on all five tasks; forecast effects ≤ 0.03% of native error. Geometric fidelity
  at the edited layer is not a proxy for downstream effect. **Rung 4 does not imply rung 5.**
- **Learned edits beat matched random only offline, not behaviourally.** Offline, the learned edit beat matched random
  by 1.2–1.5% of forecast error; on task outcomes it did not, and the +6.25 pp development gain was 0.00 pp on fresh
  scenarios. A spline must be compared with a random smooth curve of matched length *and* with the reflected and
  projected arms, and the comparison must be on data untouched by every design decision.
- **Edits were ~99% a shared constant shift across clips.** If that holds here, a spline path and a straight path
  differ by a small clip-independent correction. Measure the shared-vs-specific split of every edit first.
- **Rank matters, but "rank 1 fails" was too strong.** Rank 1 passed the effect and random gates; the sweeps that
  seemed to need "all patches" were rank-1 sweeps. What stands: effective dimension of the delivered edit matched
  random, and the edit basis contained its own readout (0° principal angles), two dimensionality readings that were
  artefacts. Check the steering basis against the evaluation probe's weights before reading any dimension number.
- **The only readout that survived scrutiny** was agreement with the model's own forward pass on *real* clips at the
  target value, tracked through later layers; probe readouts at the edited layer agreed downstream in only 41–58%
  of cases. A one-block replacement can also be inconsistent with the other blocks: edit all 8 time steps, and
  compare against a real clip at θ\* at every later layer.

**What it found about circular structure (transfers as machinery, not as a result):**

- jepa_steering *did* have cyclic variables, twice, both in archived threads. A 5 Sep screen on the JEPA-WM
  Reach-Wall predictor decoded the hand's motion direction: R² 0.046 in the image encoder rising to 0.573 at
  predictor block 3, about 10 orthogonal probe directions before chance, direction ~75° from the magnitude
  subspace. Never steered, deferred the same day. A 2 Sep loop built a circular chart X ≈ μ + A[cos θ, sin θ]
  (phase decoded by atan2 on the fitted plane), a rotating-code detector (local-mean vs global-mean projection,
  because the global mean of a code that rotates with angle is the ring's centre, ≈ 0), and a **planted rotating-code
  positive control that passed**. The real data showed no rotating code there, on two bearings per scene.
- So: the closed-spline, intrinsic-angle and chord-through-centre reasoning still comes from the physics paper and
  Goodfire. What we inherit is (i) the circular chart as the supervised check on Goodfire's unsupervised
  atan2(PC2, PC1), (ii) the global-mean-blindness argument (a single mean-difference "direction" vector is the wrong
  baseline; steering must be target-conditioned), (iii) the planted-ring positive control before any "no ring" or
  "no curvature" claim, and (iv) a prior that in a JEPA-WM predictor direction needed ~10 directions and sat ~75°
  from magnitude, to compare with the encoder here, not to import.

**What it did not find (does not transfer):**

- Nonlinear *probes* (kNN, RBF kernel ridge, bilinear, Jacobian-local) rarely beat linear ones (6/324 entries,
  chance ≈ 16), and a coordinate-transform null showed an RBF readout gain can be a frozen linear readout plus a known
  frame conversion. That is the Cartesian-vs-polar question here: nonlinear decodability of θ may just be
  atan2(vx, vy).
- It steered a predictor's forecast with a real behavioural output. V-JEPA's encoder has none, so the behaviour
  manifold must be constructed (§7).
- The four-symmetric-point curvature test could not see third-order structure; with 64 dense knots a leave-one-out
  test cannot either (the chord misses the arc by 0.5% of the radius). The knot-subsampling test in
  `nonobvious_components.md` §7 replaces both.

## 6. What "meaningful held-out steering evaluation" means, in one place

Three leaks, three separations.

1. **Leak: the spline memorised the 64 centroids.** A spline through all values evaluated on those values is
   interpolation by construction. Separation: hold out every 4th label value (interior, so the loop and the curve
   are never extrapolated), fit knots on the rest, steer toward the held-out values.
2. **Leak: the readout is the intervention.** A probe whose weights lie in the steering subspace reads back whatever
   was written there. Separation: the evaluation probe is fitted on clips disjoint from both the knots' clips and the
   steered clips, and its weight vector is checked for overlap with the steering basis.
3. **Leak: the edit looks right at layer L and nowhere else.** Separation: compare the steered activation with
   V-JEPA's own activation on a real clip at the target value, at L and at every later layer, and through the predictor
   if run. Report the layer at which agreement decays. This is the readout the talk should lead with.

Plus dose–response along the path (K waypoints, monotone readout expected), both steering directions, and the
off-target check (steer direction, read speed; steer speed, read direction). A steer that moves the off-target
variable has left the manifold even if the on-target readout is perfect.

## 7. What plays the behaviour manifold for an encoder

Goodfire's M_y is the output-probability manifold. V-JEPA's encoder emits no distribution, but Goodfire's own
Mountain Car section (§5 Eq. 9, B.1) defines one for a visual world model: a softmax over negative L2 distances
(τ = 0.5) to 128 points along the fitted spline, in PCA-64. That is what Part 2 uses for behaviour energy and isometry,
with the caveat that at the steered layer it restates the activation geometry (circular), so it is informative only
downstream. The other candidates, weakest first:

1. evaluation-probe readouts on real clips (a probe again; only for the isometry plot, never as "behaviour"),
2. nearest-real-clip agreement across later layers (the propagation readout above),
3. the predictor's forecast of masked future tokens, compared with the real future tokens of clips at the target value
   (the only candidate that is a behaviour; needs the GPU session).

State which one is used for each figure. Do not call a probe readout a behaviour.

## 8. The decision tree, so the work stays small

- Run the cheap geometry checks first (`nonobvious_components.md` §7, minutes on stored activations). They decide
  whether there is anything for a spline to do:
  - centroid loop closed and in top PCs, chord/arc gap large near 180°, within-value spread smaller than the bend →
    fit the periodic spline, run the held-out steer, expect a spline advantage that grows with angular shift;
  - loop present but spread larger than bend → the spline fits noise; report that and stop at the geometry figure;
  - speed curve straight and evenly spaced → no spline for speed; one sentence and a figure.
- Measure the Part 1 linear steer's off-manifold energy before fitting any spline. If it is already low, the spline
  has no room and the honest result is "linear is enough at this layer for this variable".
- Whatever the spline does at layer L, run the propagation test. A spline that wins at L and loses at L+3 is the
  jepa_steering result again, and it is worth one slide, not a rescue attempt.

## 9. What would count as a good Part 2 outcome

Any of these, reported with controls, is a finding:

- Spline beats line, the gap grows with angular shift, survives the random-curve and BF16 controls, and later layers
  agree: the ring is causally used and curved edits are the right tool for cyclic variables.
- Spline beats line at L but later layers do not care: geometry is real, the model routes around it; consistent with
  jepa_steering; the right null result to bring to an interpretability group.
- Spline and line coincide because the linear edit already stays on-manifold: the paper's multi-probe subspace is
  sufficient, and "40 dimensions" is enough to span the ring. Also a clear answer.

The bad outcome is a spline that wins only against a straw-man line, scored by its own probe, at one layer. Every
control in §5 and §6 exists to rule that out.

## 10. Where this sits relative to the linear representation hypothesis

The linear representation hypothesis (LRH: concepts are directions, approximately orthogonal, and moving along a
direction changes the concept) is the premise behind linear probes, sparse autoencoders and add-a-vector steering.
Goodfire's paper is framed as a test of its steering corollary; Part 2 inherits that framing. The claim should be
made precisely, because the weak form of the LRH survives here and the strong form is what is under test.

**What survives.** Direction is linearly decodable through (sin θ, cos θ). Each of those is a direction in
activation space, so a ring is a *two-dimensional linear subspace* with a nonlinear constraint on it (unit norm in
the right coordinates). Speed's curve, if it is a bent 1-D curve, likewise lives in a low-dimensional linear subspace.
The physics paper's INLP result ("tens of dimensions") is a statement that the linear subspace is larger than one
direction, not that the code is non-linear. Nothing in Part 1 refutes the LRH in its subspace form, and an honest
talk should say so.

**What does not transfer 1:1.**

- *Add-a-vector steering assumes the data manifold is flat inside the subspace.* On a ring it is not: adding a
  vector to move from θ to θ + 180° passes through the origin of the (sin, cos) plane, where the state is a
  superposition of opposite directions, not an intermediate one. Goodfire's video example shows exactly this: the
  linear path between two car positions decodes to a blurred car, "an incoherent superposition of positional
  beliefs", while the manifold path decodes to a car moving. The LRH's *representation* claim can hold while its
  *steering* claim fails.
- *Continuous physical variables have no discrete feature dictionary.* In LLMs the LRH is usually cashed out as sparse
  features (SAE latents) that are on or off. Direction, speed and position are continuous and dense: every clip has
  some value, and neighbouring values have neighbouring codes. jepa_steering found the same thing in a JEPA
  predictor: a dense curved manifold in which cubic reconstruction beat linear by orders of magnitude and no sparse
  axis-aligned basis emerged. "Which feature is speed" is the wrong question; "what is the shape of the speed
  curve" is the right one.
- *The circular and curved codes are not a JEPA peculiarity.* Engels et al. (2024) found the same for days and months
  in LLMs, Kantamneni & Tegmark (2025) a helix for numbers. The axis that matters is not language model vs world
  model but *discrete token identity vs continuous latent variable*. World models are simply dominated by the second
  kind, so the strong LRH is a worse default for them than for a token-prediction model.
- *Multi-frequency codes are still linear subspaces.* If direction's ring turns out to occupy more than two
  dimensions (higher harmonics, sin 2θ and cos 2θ, or a great circle tilted through several PCs) that is a
  multi-dimensional feature in Engels' sense: linear in the subspace, non-linear as a manifold. The INLP sawtooth
  period tells us whether the pairs are all first-harmonic (period 2 in K) or mixed.

**What this project can and cannot say about it.** Part 1 measures the subspace (rung 2) and whether writing into it
with a linear least-squares move works at the layer (rung 3). Part 2 measures whether the move has to follow the
curve (rung 4) and whether later layers agree (rung 5). The outcomes map onto the LRH as follows:

| Result | Reading |
|---|---|
| Linear multi-probe steer already stays on-manifold and propagates | strong LRH adequate for these variables at this layer; the ring is small enough that chords ≈ arcs |
| Spline beats line, gap grows with angular shift, propagates | LRH representation claim holds, steering corollary fails; geometry is causally used |
| Spline beats line at L, later layers indifferent | LRH steering corollary fails in both forms; the model routes around the edit (jepa_steering's result) |
| No curvature above noise | subspace LRH fully adequate; the question was not testable on this data |

The honest headline, if the middle rows land, is not "the LRH is false for world models" but "for continuous
physical variables the LRH describes the subspace and misdescribes the moves, and V-JEPA is a clean place to see
it because direction is cyclic by construction."
