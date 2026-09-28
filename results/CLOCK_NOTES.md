# Time-rescaled clips: a frozen-probe linearity / extrapolation test, not a clock test

**Result.** When the true speed doubles, V-JEPA 2's frozen speed probe doubles too, as far as a linear readout with an intercept predicts: observed/expected is 0.98–1.03 inside the probe's 0.25–4 m/s training range. Above that range the probe saturates (observed/expected 0.95–0.97 over all in-frame clips). For acceleration ×4, the readout matches the linear expectation in range (0.99–1.15) and saturates above 10 m/s² (0.88–0.95). The same probe recipe on pixels plus frame differences does not read speed at all. The displacement readout gives exactly 2.00 and 4.00 by construction.

**Why this is not a clock or physical-units test (ledger #251).** At the fixed 24 fps, x(λt) is by definition the clip with speed λv (acceleration λ²a). So λ = 2 at v = 1..3 and λ = 0.5 at v = 2, 4, 6 are the same videos as paper_layout clips: 168 of the 392 clips per λ, with a max relative meanpool difference of 7e-8 at point 12. For acceleration, 56 of 392 clips per λ are shared. The design cannot tell a clock from an odometer. It must not be cited as evidence about time, seconds or units.

**Source.** `results/p5_clock_test_linearity.json` (points 9, 12, 19, 22). It supersedes `results/p5_clock_test.json` (14:45 ET), which counted cross-λ duplicate clips as independent, bootstrapped per clip, and compared ratios with 2 and 4 instead of the intercept-adjusted expectation. The `provenance.overwrites` string in the new file says "replaces … this file". That is wrong: the new results went to a separate file, and the old file was left in place.

## Method

- **Probes.** Fit on the supplied sets' train split (`splits/split_v1.json`, sha256 98e6310c…) using the wm.probes recipe, then read frozen on λ = 0.5, 1 (paper_layout) and 2.
- **Expected ratio.** Calibrate pred = a·eff + b on λ = 1 in-frame clips inside the training range. A linear readout then predicts the ratio (a·λv + b)/(a·v + b).
- **Bootstrap.** 200 draws, resampling the 56 straight-line paths (direction × start). The pooled OLS keeps each distinct clip once: 116 (velocity) and 66 (acceleration) duplicate rows dropped.
- **Script hashes.** `run_clock_test.py` sha256 551367d4…, `render_clock_stimuli.py` 822582bb…. Both were dirty at commit b8856e8.

## Velocity, speed probe (key `families.velocity.readouts.<r>.scalar`)

"in range" = in frame at λ = 1 and 2 with λv ≤ 4 m/s (99 clips, 56 paths). "All in frame" = in frame at all three λ (146 clips).

| readout | ratio λ2/λ1, in range: observed | expected | observed/expected [95% CI] | all in frame: observed/expected | pooled slope, in frame [CI] |
|---|---|---|---|---|---|
| point 9 | 1.94 [1.91, 1.97] | 1.97 | 0.98 [0.96, 1.00] | 0.95 [0.93, 0.97] | 0.72 [0.69, 0.76] |
| point 12 | 1.93 [1.91, 1.96] | 1.93 | 0.99 [0.97, 1.02] | 0.95 [0.92, 0.96] | 0.72 [0.69, 0.77] |
| point 19 | 1.89 [1.86, 1.93] | 1.83 | 1.01 [1.00, 1.03] | 0.97 [0.95, 0.99] | 0.77 [0.74, 0.81] |
| point 22 | 1.91 [1.87, 1.95] | 1.82 | 1.03 [1.00, 1.05] | 0.97 [0.96, 0.99] | 0.79 [0.76, 0.83] |
| pixels (CV R² 0.44) | 1.01 [0.96, 1.09] | 0.96 | 1.08 [1.00, 1.21] | 1.32 [1.12, 1.42] | 0.01 [−0.04, 0.06] |
| trajectory, magnitude of (vx, vy) | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |

- **Keys.** `in_train_range_lam1_2.ratio_2_over_1.{median, median_ci, expected_linear_median, obs_over_expected_median, obs_over_expected_ci}`, `in_frame.ratio_2_over_1.obs_over_expected_*` and `in_frame.pooled.slope`. The trajectory row uses `.cartesian_norm`.
- **Trajectory scalar speed probe.** It fails on the supplied set (CV R² −0.00; the same happens in the existing `p1a_speed_speed_trajectory`), so the displacement reference is the magnitude of the (vx, vy) probe.
- **Per-λ fit slopes at point 12** (`in_frame.lam*.slope`). 0.99 at λ = 0.5, 0.95 at λ = 1, 0.61 at λ = 2, where λ = 2 lies mostly above 4 m/s. That is saturation in extrapolation.
- **Direction MAE is unchanged by λ.** At point 12: 4.5° [4.0, 5.3], 3.4° [3.0, 3.8] and 3.3° [2.9, 3.6] (`in_frame.lam*.direction_mae_deg`). For comparison, pixels give 84°, 70° and 51°.

## Acceleration, acceleration probe (key `families.acceleration.readouts.<r>.scalar`)

"in range" = λ²a ≤ 10 m/s² (112 clips); "all in frame" = 242 clips.

| readout | ratio λ2/λ1, in range: observed | expected | observed/expected [CI] | all in frame: observed/expected |
|---|---|---|---|---|
| point 9 | 3.77 [3.51, 3.93] | 3.28 | 1.13 [1.02, 1.21] | 0.88 [0.85, 0.91] |
| point 12 | 3.57 [3.39, 3.70] | 3.37 | 1.04 [0.96, 1.12] | 0.88 [0.85, 0.92] |
| point 19 | 3.48 [3.37, 3.76] | 3.46 | 0.99 [0.94, 1.07] | 0.89 [0.87, 0.93] |
| point 22 | 3.35 [3.24, 3.45] | 2.81 | 1.15 [1.07, 1.29] | 0.95 [0.90, 0.99] |
| pixels | 0.98 | 1.08 | 0.92 [0.88, 0.98] | 1.05 [0.87, 1.40] |
| trajectory, magnitude of (ax, ay) | 4.00 | 4.00 | 1.00 | 1.00 |

Direction MAE at λ = 0.5 rises to 19–25° because slow accelerating clips barely move (0.25–1.75 m/s² from rest). At λ = 1 it is 4.4–6.1° and at λ = 2 it is 2.9–3.7°.

## Stimuli

- **Rendering.** `artifacts/stimuli/clock_lam{0.5,2}` and `clock_acc_lam{0.5,1,2}`, 392 clips each, from the paper_layout metadata. The λ = 1 render is bit-identical to paper_layout on 4 checked clips.
- **In-frame counts (disk fully in frame at every frame).**
  - Velocity: 382 at λ = 0.5, 288 at λ = 1, 146 at λ = 2.
  - Acceleration: 392, 392 and 242.
- **Out-of-frame clips.** Nothing was dropped; out-of-frame clips are flagged in the metadata.

## Slide p2-clock (talk/slides_v2)

The slide's numbers come from `results/p5_time_manifold.json`, not from this test. The NUMBERS.md row that cites `p5_clock_test.json` is a misattribution. Checked keys, identical at HEAD and in the working tree:

- **V-JEPA.** `clock["direction/vjepa2/timepool"]["12"].spline.{slow,mid,fast}.slope.mean` = 1.003 [0.997, 1.008], 0.998 [0.995, 1.000] and 0.996 [0.995, 0.997]. `odometer_prediction` = 0.38, 1.00 and 1.63.
- **Untrained copy.** `direction/random/timepool` gives 0.51, 0.58 and 0.94.

Ledger #251 does not directly hit this result: it varies speed at a fixed frame count, not the clock. But the slide's time wording overreaches. At fixed 24 fps, step index, frame count and elapsed time are one variable, and the per-step code may be the encoder's token-time position code (the random-init encoder decodes t at 0.97). So "elapsed time" and "keeps a clock" in the sense of physical time are not shown.

**Defensible wording.**
- Title: "The time-step code is speed-invariant: it counts frames, not distance".
- Body: "Held-out slow / mid / fast clips advance 1.00 / 1.00 / 1.00 steps per step along the time spline (a distance code: 0.38 / 1.00 / 1.63; untrained copy 0.51 / 0.58 / 0.94). At a fixed frame rate this cannot separate frame count from elapsed seconds."
- Drop the citation of `p5_clock_test.json` as a clock control. If it is kept, it should read: "Time-rescaled renders are speed changes in disguise; they test probe linearity: the speed readout doubles as a linear readout predicts inside the training range and saturates above it."
