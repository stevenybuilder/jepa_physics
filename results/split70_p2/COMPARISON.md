# Part 2 contiguous-design steering at the paper's 70/30 split vs the 80/20 split of record

Same code (HEAD 70d813d), WM_SPLIT_PATH=splits/split_paper70.json, knot folds 0–2 / probe folds 3–4 inside train / test steered, seed 0, smoothing spline, matched support, all controls. Run 2026-09-27 20:16–20:21 ET on CPU.

| variable, point | split | verdict | endpoint gap spline−chord (+ = spline better) | min readout radius gap | bend / centroid noise |
|---|---|---|---|---|---|
| acceleration L12 | 80/20 | path_geometry_positive | -0.003 | +nan | 0.07 |
| acceleration L12 | 70/30 | path_geometry_positive | -0.006 | +nan | 0.10 |
| acceleration L21 | 80/20 | path_geometry_positive | -0.017 | +nan | 0.07 |
| acceleration L21 | 70/30 | path_geometry_positive | -0.008 | +nan | 0.04 |
| direction L12 | 80/20 | path_geometry_positive | -0.116 | +0.272 | 0.36 |
| direction L12 | 70/30 | path_geometry_positive | +0.405 | +0.266 | 0.14 |
| direction L22 | 80/20 | negative_endpoint | -3.851 | +0.253 | 0.47 |
| direction L22 | 70/30 | path_geometry_positive | -0.411 | +0.276 | 0.36 |
| speed L12 | 80/20 | path_geometry_positive | -0.030 | +nan | 0.19 |
| speed L12 | 70/30 | negative | -0.012 | +nan | 0.06 |
| speed L19 | 80/20 | path_geometry_positive | -0.011 | +nan | 0.13 |
| speed L19 | 70/30 | path_geometry_positive | -0.000 | +nan | 0.06 |

Reading: the direction path result (spline keeps the readout radius, the chord collapses it: +0.25–0.28 at both points and both splits) is unchanged. The point-22 direction endpoint loss at 80/20 (−3.85°, 'negative_endpoint') shrinks to −0.41° at 70/30 and the call becomes path-geometry positive, consistent with the 16-arc sweep where 3 of 16 arcs show that loss. For speed and acceleration the bend is 0.03–0.19× centroid noise at both splits, so the verdict flickers between 'path-geometry positive' and 'negative: no curvature benefit' on sub-margin differences; the reading 'scalars are straight, no independent-readout gain' holds at both splits. Split of record stays 80/20.
