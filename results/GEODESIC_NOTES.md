# True G_E geodesic vs spline vs raw chord (direction, points 12 and 22), LITE run

**Answer (3 held-out targets x 16 carriers, per layer):** the Eq. 6 geodesic does **not** follow the direction ring, and it does **not** beat the spline on the readouts. It does beat both on the quantities it minimises: its Eq. 4 length and its distance to real clips. Its readouts sit at the chord's level.
- **Formulas (refs/steering_paper.txt).** l.1396 Eq. 4: L_G(π) = ∫ sqrt(π̇ᵀ G(π) π̇) dt. l.1398: "a geodesic is defined as the path of minimum length between two endpoints". l.1407 Eq. 6: G_E(h) = (α e^{−E(h)} + β)^{−1} I. l.1411: "α, β > 0 are calibration constants". l.1427-1428: "constants α, β > 0 calibrating the dynamic range (Béthune et al., 2025). Geodesics under GE thus follow Mh". l.1271: E(x) ∝ −log p(x). l.2731 (A.7): E_BC sums Bhattacharyya distances over "the K = 50 waypoints".
- **Calibration.** Goodfire gives no values for α and β. Béthune et al. 2505.18230 §3.3 says the constants are "chosen so that the metric scale to I on the data manifold and to 10³·I in low-density regions". So p̃ = e^{−(E−E_ref)}, where E_ref is the median leave-one-out train energy, with β = 1e-3 and α = 1 − β.
- **Energies.** Both are fit on TRAIN only: knot clips (folds 0-2) at the kept values. The held-out arc is therefore low-density by construction. The variant `geo_knnall` refits the kNN energy on knot clips at all values; probe and test clips never enter either fit.
  - kNN energy: E = d·log r̄₅, where r̄₅ is the mean full-space distance to the 5 nearest clips (the off_manifold_energy measure) and d is the Levina-Bickel intrinsic dimension: 13.1 at point 12, 7.3 at point 22.
  - KDE energy: top-10 PCs, whitened; the 3-fold held-out log-likelihood bandwidth is h = 0.58 at point 12 and 0.50 at point 22, not at the edge of the grid. These PCs contain 99.8% of the ring plane.
- **Discretisation.** 50 free nodes between fixed endpoints, Simpson quadrature on each segment, torch L-BFGS (strong Wolfe) from session2_pullback_goodfire.optimise. The path is resampled to 50 waypoints uniform in arc length and read with run_part2.evaluate.
- **Deviations (lite, for the 17:10 deadline).**
  - Only targets 303.75, 326.25 and 343.125 (of the 8), 16 carriers each, with no random-restart null.
  - L-BFGS was run for ≤100 outer steps with max_iter 5 and tol 1e-5. **Every solve stopped at the step cap, so none converged.** Treat the L_G values as upper bounds.
  - Endpoints are pinned to the raw chord's end state, so endpoint error and nearest-real R equal the chord's by construction ("pinned"). `geo_knn_spline_end` is pinned to the spline's end state instead.
- **Parity (G = I, started from the spline).** It reaches L = 1.005× the chord at point 22 and 1.025× at point 12. Its mean closest-point distance to the chord is 0.24 and 0.33, against 3.9 and 4.6 for the spline. That is close to the chord but not exact at 100 steps.
- **Path metrics, point 12** (chord / spline / kNN geodesic from the chord / KDE geodesic from the chord):

  | Metric | Chord | Spline | kNN geodesic | KDE geodesic |
  |---|---|---|---|---|
  | Min readout radius | .70 | .90 | .77 | .69 |
  | A.7 E_BC | 1.42 | 0.86 | 1.35 | 1.18 |
  | Excess distance to nearest real clips | +0.25 | +1.09 | −0.66 | 0.00 |
  | L_G under kNN energy | 14.3 | 133.6 | 8.2 | — |
  | L_G under KDE energy | 25.3 | 91.5 | — | 10.0 |
  | Euclidean length / chord | 1 | 3.49 | 1.56 | 1.39 |

  - Geodesic minus spline, radius and E_BC: the per-target 95% CI excludes 0 in all 3 targets, favouring the spline.
  - kNN geodesic minus chord: CIs overlap 0.
- **Path metrics, point 22** (same arms):

  | Metric | Chord | Spline | kNN geodesic | KDE geodesic |
  |---|---|---|---|---|
  | Min readout radius | .71 | .88 | .70 | .68 |
  | A.7 E_BC | 4.04 | 3.76 | 4.06 | 4.03 |
  | Excess distance to nearest real clips | +0.54 | +0.56 | −0.27 | +0.32 |
  | L_G under kNN energy | 21.4 | 33.9 | 16.9 | — |

- **Ring following.** Measured on the geodesic's bend (its deviation from the chord):
  - In-plane share: 0.08-0.13 of the bend lies in the ring plane (0.10-0.22 from the spline initialisation), against 0.13-0.35 for the spline and 2/64 = 0.03 for a random direction.
  - Cosine with the spline's bend: −0.00 to 0.08 from the chord initialisation, 0.10-0.35 from the spline initialisation.
  - Closest-point distance: 3.1-4.1 to the spline and 1.8-2.7 to the chord.
  - Under G_E, the spline is 1.4-9.4× *longer* than the chord, so the density metric does not prefer the ring route.
- **Multimodality.** The initialisations disagree: L_G differs by more than 1% in 25-69% of kNN paths and 50-100% of KDE paths, and the two geodesics lie 0.8-2.2 apart. Because no solve converged, this does not separate a multimodal landscape from an unfinished optimisation.
- **Refit on all values.** It barely changes the route: the geodesic lies 0.73 (point 12) and 0.87 (point 22) from the train-only geodesic, with ring-plane share 0.07-0.08.
- **Files.**
  - results/p2_geodesic_direction_L{12,22}.json: merged from per-target shards; shard sha256 values are in the provenance block.
  - figures/fig_geodesic_direction_L{12,22}.png: one carrier, target 303.75.
  - scripts/run_geodesic.py and tests/test_geodesic.py.
