# Spline validation (Part 2, direction): verdict

**Verdict.** The smoothing level is well calibrated and the held-out logic is sound. Two problems remain. First, the FITPACK smoothing spline is not invariant to the arbitrary zero and orientation of the atan2 angle. Second, issue #213 misaimed the chords. Fixing the aim widens the endpoint gap between the spline and the raw chord at point 12. Numbers: `results/p2_spline_validation.json`.

| Check | Pass | Key number |
|---|---|---|
| Periodic parameterisation / wrap | ✓ | per=1 with the knot repeated at t0+2π; evaluation wraps from t0 |
| Knot order (label-free angle) | ✗ | 8/16 point-12 arcs have 1–3 foreign knots between a target's value neighbours |
| s = m, w = √n / sd | ✓ | leave-one-knot-out picks s = m on 17/17 point-12 arcs (error 3.10 vs 3.38 at 0.5m, 3.56 at 2m) |
| Invariance of the smoothing spline | ✗ | rotating or reflecting the angle moves held-out targets by up to 4.5× centroid noise (headline arc) and 10.8× (seed 7); the interpolating spline moves 0 |
| Interpolating alternative | ✓ rejected | LOKO error 10.97; the curve runs 117 PCA units from the knots |
| Cyclic coordinate | ✓ | label-free at point 12 (16/16 arcs); labels at point 22 |
| Held-out aim | ✗ → fixed | the legacy chord was misaimed by 2.3–3.8 PCA units; now `--aim arc` |
| Scalar extrapolation | ✓ | end-tangent continuation |
| Held-out logic / disjointness | ✓ | knot folds 0–2 at kept values; probe 3–4; carriers are test clips |
| Probe readout | ✓ with a caveat | a linear sin/cos probe reads a chord point at the correct angle, so the endpoint error cannot reward curvature |
| Controls | ✓ | random, shuffled, dose-matched, projected and reflected arms |
| CIs across arcs | ✗ | mean ± SD only; the arcs are not independent; seed 7 dominates the mean |
| Verdict margins | ✗ | the endpoint margin is the probe's out-of-sample error (~3.9°), so a +2.4° loss never counts; the comparator is the smoothed chord, not the A.9 raw chord |

**#213 fix.** `Curve.aim="arc"` places a held-out target at its value fraction of arc length between the two knots that neighbour it in value. The spline is walked in angle order. Both chords join the value neighbours. The default `aim="coord"` is the old code path, byte for byte. Tests: `tests/test_manifold.py` (a synthetic non-monotone ring).

**Point 12, spline − raw chord, 16 arcs** (endpoint probe error in °, nearest-real R):

| Set | Endpoint gap | Spline ahead | Nearest-real gap |
|---|---|---|---|
| Published (Mac, legacy code) | +1.32 ± 1.77 | 3/16 | −0.021 ± 0.061 |
| Requested: 8 clean arcs as stored + 8 fixed | +4.91 ± 9.76 | 1/16 | −0.026 ± 0.077 |
| Same, without seed 7 | +2.51 ± 1.70 | 1/15 | −0.01 ± 0.04 |
| Same box, legacy → fixed, without seed 7 | +1.25 ± 1.55 → +2.41 ± 1.68 | 3/15 → 0/15 | −0.005 → −0.003 |

On the eight affected arcs the endpoint gaps before → after the fix are:

| Seed | 1 | 4 | 5 | 7 | 8 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|
| Gap before → after (°) | −1.65 → +2.09 | +3.97 → +6.05 | +0.17 → +2.11 | −1.43 → +40.97 | +3.96 → +5.82 | +0.79 → +2.22 | +0.67 → +2.55 | +0.50 → +2.64 |

Seed 7 is an invariance failure, not an effect of the aim fix. With the same legacy code and the same data, the spline's endpoint error is 9.1° on the Mac and 52.8° on the box. The PCA sign conventions differ between the two machines, and FITPACK's adaptive knot placement then produces a different spline. The raw chord is identical on both machines. The verdict holds at "path_geometry_positive" on 15 of 16 arcs; seed 7 now reads "negative_endpoint".

**Figures.** The old p2-ring plot does misrepresent the fitted curve. It draws the *interpolating* spline, fit on all train clips, through label-free knots with 6 neighbour swaps and two knots 0.01° apart, and it overshoots to 16.7 PCA units against a ring of radius about 6.7. The steering curve never loops. At point 22 the top-2 PC plane of the centroids shows ring plus fold, which draws the ring as a figure 8. The new `figures/fig_ring_fitted_L{12,22}.png` plot the fitted steering spline (4,000 points in angle order) in the ring plane.

**What a careful reviewer would reject:** (1) the spline depends on the platform through FITPACK knot placement. The fix is a Reinsch smoother with a knot at every site (as in Goodfire's `CubicSpline1D`), with λ chosen by LOKO. (2) The published point-12 raw-chord mean was computed with misaimed chords. (3) The endpoint margin is too lenient to register an endpoint loss. (4) There is no inference across arcs.
