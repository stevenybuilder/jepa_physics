# V-JEPA 2 physics take-home

This repository probes and steers the frozen V-JEPA 2 ViT-L/16 encoder on the supplied disk clips. It repeats the
layer-wise probing, nullspace and multi-probe steering experiments of Joseph et al. (arXiv 2602.07050), with a partial
reproduction: the pooled Physics Emergence Zone does not appear on the supplied clips, the direction-versus-speed
probe-count contrast does not hold, and the steering dose-response reproduces with the paper's Adam basis only. It
extends them with
spline (manifold) steering after Wurgaft et al. (arXiv 2605.05115). The short version: readable is not the same as used.
Edits reach the predictor's forecast late in the encoder, or earlier only through the object's own tokens. A curved
edit steers the forecast's heading code, while a straight edit of the same size moves the whole forecast further.

The full write-up is [REPORT.md](REPORT.md). The 15-minute talk is at https://claude.ai/artifact/JdPJ1Yig3LpL5KntZCJo88
(deck sources in `talk/`).

## What the take-home asked, and where each ask is answered

| Instruction | What we did | Scripts | Results | REPORT |
|---|---|---|---|---|
| Part 1.1 Layer-wise probing | Ridge probes at all 26 read points for direction, speed and acceleration, with onsets, an untrained copy and per-patch probes; layer curves in `figures/fig1_layer_curves.png`, sizes in `figures/fig_scaling_layerwise.png`. | `run_step1.py`, `extract.py` | `results/p1a_*` | §3.1 |
| Part 1.2 Iterative nullspace probing | INLP at the paper's layer, with nested and paper-protocol counts, an Adam rerun and a whitened count. | `run_step2.py`, `run_step2_dims.py` | `results/p1b_*` | §3.2 |
| Part 1.3 Multi-probe steering on held-out data | Steering in the span of the first N probes, on test clips never used to build the basis, against random-basis nulls; the paper's judge is a probe fit in-sample on the steered test clips, so we also report a split-half judge. | `run_step3.py` | `results/p1c_*` | §3.3 |
| Part 2: splines for speed, acceleration, direction | Periodic cubic splines through class centroids in a PCA subspace, plus lines and a direction × speed sheet. | `run_part2.py`, `run_velocity_sheet_predictor.py` | `results/p2_steer_*`, `results/p5_velocity_sheet_*` | §4.1, §4.3 |
| How splines are built, shown and evaluated | Built on knot clips only; plotted as rings, lines and sheets; scored at held-out targets, along the path and through the predictor. | `run_bakeoff_unified_16arc.py`, `run_session2.py` | `results/p2_bakeoff_unified_16arc.json`, `results/session2_*` | §4.2, §4.5 |
| The circular structure of direction | Direction lies on a ring; we test its angle coordinate and which frame describes it best. | `run_geometry_checks.py`, `run_coordinate_competition.py` | `results/p2_geometry_*`, `results/p5_coordinate_competition.json` | §4.1, §4.4 |
| A meaningful held-out steering evaluation | A whole 45° arc of directions is held out of spline construction, on 16 seeds (15 arcs); readers are fit on separate probe clips; the predictor's forecast and rendered twins serve as judge and ceiling. | as above | as above | §4.2, §7.1 |
| Comparison with the multi-probe method | Six steering arms on one arc set, including the Part 1 probe subspace, plus predictor-level tests. | `run_bakeoff_unified_16arc.py` | `results/p2_bakeoff_unified_16arc.json` | §4.3, §4.4 |
| Encoder frozen; extraction and pooling documented | No weights are updated. Activations are the mean over all 2,048 tokens (8 × 16 × 16) at each point, 16 frames at 256², no crop. | `src/wm/extract.py` | `artifacts/` (not in git) | §2 |
| Fit data separated from evaluation data | One stratified 80/20 split per dataset; probe fitting, layer choice, nullspace and spline construction use train folds only; Part 2 uses knot folds 0–2, probe folds 3–4 and the held-out test fold. | `make_splits.py` | `splits/split_v1.json` | §2, §4.2 |
| Supplied data unchanged; artifacts stored elsewhere | The data is read in place; derived files live in `artifacts/`, `results/` and `figures/`. | | | §8 |
| Deliverable: a 15-minute presentation | The talk above, with speaker notes and a number ledger (`talk/slides_v2/NUMBERS.md`). | `talk/` | | |

## Part 1: what we did and found

**Probing.** On the supplied clips direction is already readable after one block (pooled R² 0.875), and speed and
acceleration are readable from block 1 too (REPORT §3.1). So the paper's Physics Emergence Zone does not appear in the
pooled readout here. What training adds is a per-patch direction code: V-JEPA 2 reaches a mean per-position R² of 0.96 by
block 6, while an untrained copy never exceeds 0.39. On a harder rendered set the zone does appear, as a handover
(means over three render seeds): a direction code that transfers across the two halves of the frame at block 1 (0.71)
stops transferring by block 8 (−1.43; at block 4 two of the three seeds still transfer, +0.16 / +0.15), recovers only to
about chance at block 9 (0.18) and transfers durably later (0.58 at block 12).

**Nullspace.** At block 9, 37 probes are needed before direction is at chance (46 under the paper's protocol), against 39 for speed and 41 for acceleration. After
whitening, one two-output probe does the job, so the count mostly reflects the shape of the covariance, not an intrinsic
rank. (REPORT §3.2).

**Steering.** With our ridge basis, five probes steer to 8.7° from the target, as judged by a probe fit in-sample on the steered test clips (14.7° under a split-half judge). That beats a same-rank random subspace
(54.9°, p 0.035) but not the full-rank random basis (11.8°, p 0.22). With the paper's Adam probe sequence, 18 probes are
needed to reach 10° (REPORT §3.3).

**Acceleration.** In the supplied set every clip starts at rest, so acceleration, mean speed and displacement are
identical by construction. On a rendered grid that decorrelates them, signed acceleration appears in pooled features
at block 8 and is about zero there at blocks 1–4, while time-ordered per-tubelet features read it from block 1 (0.53 at block 1, 0.50 at block 4). The magnitude |a| is weak everywhere (REPORT §2, §5).

**Model size.** ViT-L, ViT-H and ViT-g all read direction in the first tenth of depth (onset 0.083, 0.094 and 0.10 of
depth). Untrained copies read 0.85–0.87 from block 1, so the early onset is mostly architecture. On the harder render (ViT-H only, one seed) the zone moves earlier as a fraction of depth by its first recovery and later by its durable recovery, so its size dependence is unsettled (REPORT §3.5).

## Part 2: what we did and found

**Construction.** Our default is a smoothing spline through class centroids in a 64-dimensional PCA subspace (the
paper's B.1 vision-model recipe), with the paper's A.3 interpolating periodic cubic spline run as a separate arm; the
headline 11.3° through the predictor below is the interpolating arm, and our smoother gives 44.2° there, not because of its size but because on that arc it already misses in the encoder (10.7° against 3.6° for the chord). We also fit a knot-cross-validated smoother and straight edits (the raw
chord between centroids, and a straight edit in cos θ, sin θ, cos 2θ, sin 2θ). Speed and acceleration lie on lines, so
splines add nothing there. Direction lies on a ring (REPORT §4.1).

**Evaluation.** Each held-out arc of 8 directions is never used to build a curve. Readers are fit on separate probe
clips. We score endpoint error and the readout radius along the path (REPORT §4.2).

**Endpoints and paths.** At held-out endpoints the raw chord lands closer than the paper's spline (0.8° at block 22)
and our smoother (1.5° at block 12, 2.3° at block 22, over 16 seeds (15 arcs)); only the exploratory knot-cross-validated smoother
edges it, by 0.4–0.5°. The curves win on the path: every curved arm keeps a higher readout radius, while the chord cuts
across the ring's hollow (forced for any straight edit between near-opposite headings, so not evidence on its own). The
evidence is the order: going either 180° route to the antipode at block 22, our smoother's readout on an independent MLP
rises steadily from 2° to 178° along the path, while the raw chord's stays within 11° until halfway and then jumps to
155–180° (REPORT §4.1, §4.3).

**Through the predictor.** Edits at block 12 spread over the whole frame are repaired by the encoder (79.7° from target
against 92.1° unedited at natural size). The same edit on the disk's own tokens reaches the forecast (44.7°). At block
22 the spline's forecast heading is much closer to the target than the chord's (11.3° against 27.2° at own size).
Without the heading probe the picture is mixed: at matched size the chord moves the whole forecast further (recovery
0.234 against 0.186), and the spline only keeps a small twin-identification lead (REPORT §4.5). Token patching shows the
direction the forecast carries comes from the disk's tokens through block 12 (0.88) and from the whole frame by block 22
(0.22 from the disk) (REPORT §4.6). Almost every attention head puts high density on the disk's tokens, but at the ablated blocks the eight with the highest previous-slot density matter no more than random heads of the same number (0.040 against 0.124–0.220 of R²), a null; of the nine heads that prefer the previous slot under a strict criterion, the three strongest sit in blocks that were not ablated, and the paper's local-attention masking test was not run (REPORT §4.7).

**Other variables.** A joint direction × speed sheet beats composed one-dimensional edits on forecast direction (29.8°
against 34.5°) but not a direction-only ring edit (28.2°) (REPORT §4.4). A "contact" edit writes a post-contact heading,
not a contact: a heading edit with no wall turns the forecast further (0.77 against 0.48) (REPORT §4.5). The within-clip
time code tracks the token slot, not the frames' content (slot coefficient 0.956), and a push along it does not advance
the forecast at block 22 (REPORT §5).

**Failure cases.** The Part 1 probe subspace at the chord's size misses by 22.8° at block 22. The paper's spline fails on
our label-free point-12 angle (34.5°). The straight Fourier edit is slightly worse than the chord through the predictor
(+2.9°), and extrapolates in the encoder but not through the predictor (32.2° against 22.2° for the spline continued
along its tangent). Conceptor steering and energy geodesics add nothing over the chord (REPORT §4.4).

## Comparison with the multi-probe method

The probe subspace is simple and needs no curve. It lands close at held-out endpoints (4.12° at block 12 over 16 seeds (15 arcs)),
but its tie with the chord at block 22 uses a 1.5× larger edit, and at equal size it misses by 22.8° (REPORT §4.3).
Splines follow the ring, so they keep the readout on the manifold along the path and steer the forecast's heading
better through the predictor. They need a trustworthy coordinate, though: on a label-free angle the paper's spline
overshoots off the ring. Neither method moves the whole forecast most of the way to the real counterfactual, and both
are repaired at mid-depth unless the edit is placed on the object's tokens (REPORT §4.4, §4.5).

## Limitations

One stimulus family, one frame rate and one render. The predictor results read the forecast's heading code with a
probe, not a rendered future. Some follow-ups are small (48 to 200 carriers, or one arc) and are flagged in REPORT §7.1.
The session-2 cache of twin forecasts does not match a recompute, and this is unresolved (REPORT §7). The acceleration
grid covers only |a| ≤ 3 m/s² and cannot separate acceleration from initial speed. The Physics Emergence Zone reproduces
only on the harder render, not on the supplied clips.

## Repository layout and how to reproduce

- `src/wm/`: library code (extraction, probes, INLP, steering, manifolds, the predictor readout).
- `scripts/`: one entry point per experiment. Part 1: `extract.py`, `make_splits.py`, `run_step1.py`, `run_step2.py`,
  `run_step3.py`. Part 2: `run_part2.py`, `run_bakeoff_unified_16arc.py`, `run_session2.py`,
  `session2_direction_natural_norm.py`, `session2_disk_token_sweep.py`, `run_token_patching.py`,
  `run_coordinate_competition.py`, `run_accel_grid.py`, `run_scaling_layerwise.py`. Figures: `make_figures.py`.
- `tests/`: run with `.venv/bin/python -m pytest -q`.
- `results/`: every number, as JSON; 313 of the 334 files carry a provenance block (split hash, commit, dirty flag).
- `figures/`: plots used in the report and talk.
- `talk/`: deck sources and `NUMBERS.md`.
- `splits/split_v1.json`: the train/test split.
- `artifacts/`: activations and other large derived files (git-ignored).
- The supplied data stays read-only in `vjepa-physics-takehome-4E00/data/`.
