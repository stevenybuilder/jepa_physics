# Part 2 contiguous-design steering at the paper's 70/30 split vs the 80/20 split of record

Same code (HEAD 70d813d), WM_SPLIT_PATH=splits/split_paper70.json, knot folds 0–2 / probe folds 3–4 inside train / test steered, seed 0, smoothing spline, matched support, all controls. Run 2026-09-27 20:16–20:21 ET on CPU.

Values are `gaps.manifold_minus_linear.{probe_err_to_target,probe_radius_min}.mean` with `ci95` from `results/p2_steer_*_contiguous.json` (80/20) and `results/split70_p2/p2_steer_*_contiguous.json` (70/30), the fields REPORT.md §3.4 and §4.3 quote. Endpoint error is in degrees for direction, m/s for speed, m/s² for acceleration. Verdict is `verdict.call`; bend is `verdict.sagitta_over_noise_median`.

| variable, point | split | verdict | endpoint probe error, spline − chord [95% CI] (+ = spline worse) | min readout radius, spline − chord [95% CI] (+ = spline better) | bend / centroid noise |
|---|---|---|---|---|---|
| acceleration L12 | 80/20 | path_geometry_positive | +0.004 [-0.001, 0.009] | n/a (scalar) | 0.07 |
| acceleration L12 | 70/30 | path_geometry_positive | +0.006 [0.004, 0.008] | n/a (scalar) | 0.10 |
| acceleration L21 | 80/20 | path_geometry_positive | +0.016 [0.009, 0.022] | n/a (scalar) | 0.07 |
| acceleration L21 | 70/30 | path_geometry_positive | +0.008 [0.004, 0.012] | n/a (scalar) | 0.04 |
| direction L12 | 80/20 | path_geometry_positive | +0.114 [0.049, 0.176] | +0.256 [0.224, 0.290] | 0.36 |
| direction L12 | 70/30 | path_geometry_positive | -0.400 [-0.473, -0.330] | +0.252 [0.224, 0.283] | 0.14 |
| direction L22 | 80/20 | negative_endpoint | +3.800 [3.358, 4.231] | +0.239 [0.207, 0.275] | 0.47 |
| direction L22 | 70/30 | path_geometry_positive | +0.467 [0.259, 0.688] | +0.264 [0.234, 0.293] | 0.36 |
| speed L12 | 80/20 | path_geometry_positive | +0.030 [0.024, 0.037] | n/a (scalar) | 0.19 |
| speed L12 | 70/30 | negative | +0.012 [0.009, 0.015] | n/a (scalar) | 0.06 |
| speed L19 | 80/20 | path_geometry_positive | +0.012 [0.008, 0.015] | n/a (scalar) | 0.13 |
| speed L19 | 70/30 | path_geometry_positive | +0.000 [-0.001, 0.001] | n/a (scalar) | 0.06 |

Reading: the direction path result (spline keeps the readout radius, the chord collapses it: +0.24–0.26 at both points and both splits) is unchanged. The point-22 direction endpoint loss of the spline at 80/20 (+3.80°, 'negative_endpoint') shrinks to +0.47° at 70/30 and the call becomes path-geometry positive, consistent with the 16-arc sweep where 1 of 16 arcs on the all-labels set shows that loss (3 of 16 on the mixed-angle stored set). At point 12 the endpoint gap is +0.11° at 80/20 and −0.40° (spline slightly better) at 70/30, both well under the probe's out-of-sample error. For speed and acceleration the bend is 0.04–0.19× centroid noise at both splits, so the verdict flickers between 'path-geometry positive' and 'negative: no curvature benefit' on sub-margin differences; the reading 'scalars are straight, no independent-readout gain' holds at both splits. Split of record stays 80/20.
