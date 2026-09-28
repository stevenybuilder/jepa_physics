# Visible numbers in slides_v2

The cites slide shows paper years only, from papers.md → "Slide citations".

Every number shown on a slide, with the file and key it was read from. Numbers inside figures are drawn from the same files by `talk/make_talk_figs.py`. Speaker-note numbers are sourced in each `<aside>`.

| Slide | Number on slide | File → key |
|---|---|---|
| p1-training | 0.875 | results/p1a_direction_direction_meanpool.json → layers[1].cv_mean (0.8750) |
| p1-training | 0.848 | results/p1a_direction_direction_meanpool_random.json → layers[1].cv_mean (0.8480) |
| p1-training | 0.96 | results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 6 (0.9572) |
| p1-training | 0.38 | results/p1a_perpatch_direction_random.json → max(curves.perpos_mean_r2) (0.3845); REPORT §1 rounds this to "never exceeds 0.39" |
| p1-zone | negative, blocks 4–8 | results/p1a_perpatch_hard_seeds.json → by_point[point 4, 6, 7, 8].cross_half_r2.mean (−0.136, −0.362, −1.082, −1.432) |
| p1-zone | 9 | same file → by_point[point 9].cross_half_r2.mean (+0.181), the first positive point after 8 |
| p1-zone | 4–6 | same file → onsets.{0,1,2}.perpos_mean_r2.from_point = 4, to_point = 6 |
| p1-anisotropy | 37 | results/p1b_direction_direction_meanpool_L9.json → K |
| p1-anisotropy | 3.2° | results/p1c_direction_L9_rank2_covweighted.json → edits.covweighted_probe1.mae_to_target (3.2013) |
| p1-anisotropy | 0.03 m/s | same file → edits.covweighted_probe1.off_target.mean_abs_change (0.0335) |
| p1-anisotropy | 4.2 m/s | same file → null_medians.rows.ridge_alpha_100_stored.covweighted_probe1.off_target_speed_mean_abs_change_median (4.1778) |
| p2-ring | −0.98 | results/p2_geometry_direction_L12.json → angle.circular_corr (−0.9832) |
| p2-radius | 16 of 16 | results/arcs_rawchord/L12_s{1..16}/p2_steer_direction_direction_L12_contiguous.json → gaps.manifold_minus_linear_raw.probe_radius_min.mean > 0 on 16 of 16 (mean +0.261 ± 0.062). Point 22: 16 of 16, with the L22_s{4,8,9,10}_labels runs substituted on the four arcs where the label-free angle fails (+0.279 ± 0.046) |
| p2-radius | 0.23 | results/p2_steer_direction_direction_L12_contiguous_rawchord.json → min(waypoint_readout.linear_raw[3].radius) (0.2305, shift bin 135–180°) |
| p2-compare | 4.2° (multi-probe) | results/p2_bakeoff_direction_direction_L12_contiguous_rawchord.json → arms.probe_qr.err_probe_unmatched (4.2466; headline arc, own edit norm) |
| p2-compare | 4.7° (chord) | same file → arms.chord_raw.err_probe_unmatched (4.7116); matches REPORT §1 |
| p2-compare | 9.7° (spline) | same file → arms.spline.err_probe_unmatched (9.7312); matches REPORT §1 |
| p2-compare | 0.16 m/s | results/p1c_direction_L9.json → single[n=5].off_target.mean_abs_change (0.160) |
| p2-compare | conceptor endpoint 165°, radius 0.70, speed 15 m/s | results/p2_conceptor_direction_L12.json arms.coast_a.{endpoint_err_deg.mean 165.03, probe_radius_min 0.699, off_target.abs_change_speed_mps.mean 15.24}; chord/spline radius arms.{linear_raw,manifold}.probe_radius_min 0.629 / 0.862 |
| p2-compare | off-target speed 0.12 / 0.43 / 0.19 m/s | results/p2_offtarget_direction_L12.json arms.{probe_qr,chord_raw,spline}.speed.mean 0.121 / 0.431 / 0.187; natural_spread.speed_mps 0.140 |
| p2-compare | notes: block-22 conceptor 41.8° [38.9, 44.5]; AND invalid 96% | results/p2_conceptor_direction_L22.json arms.coast_a.endpoint_err_deg; L12 aperture.pinv_AND_invalid_frac_by_alpha["0.1"] 0.964 |
| vs-paper | 4.7° vs 9.7°, 3.6° vs 10.7°; chord lead 1.3 / 2.3 over 16 arcs; 7/8 vs 0/8 stop-rule cells; 18 probes | same sources as p2-compare, p2-radius, p1-anisotropy and REPORT §3.2 (results/p1b_sameclip_direction_vs_speed.json) |
| p2-clock | slope 1.00/step at every speed; odometer 0.38 / 1.00 / 1.63 | results/p5_clock_test.json (see slide notes for keys) |
| p2-shapes | sheet 2.9° vs 1-D ring 6.9° at point 22; speed leakage 0.03 m/s | results/p5_velocity_sheet.json |
| p2-repair | attn −45%, MLP −40% blocks 13–16; 9% left at 24 (edit at 12); 68% (edit at 22); MLP 0.00 after 22 | results/p5_repair_attribution_L12.json attribution_summary, full.*.none.per_site |
| p2-generalise | −0.03° L12, +3.8° L22; raw chord ahead 4.3° / 4.8° | results/p2_steer_direction_direction_L{12,22}_contiguous_ctx-hard.json gaps.manifold_minus_linear{,_raw}.probe_ctx_err_to_target |
| p2-readuse | 0.976 | results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 14 (0.9763) |
| p2-readuse | 0.939 | same file → curves.perpos_mean_r2 at point 24 (0.9390) |
| p2-readuse | 11.3° | results/session2_predictor_native_readout.json → per_layer.22.spline.dir_err_to_target.mean (11.3174) |

## Corrections applied in the notes

1. **Nearest-real agreement, spline minus raw chord.** Over the 16 arcs this is −0.02 ± 0.06 at point 12, with the spline ahead on 7, and +0.02 ± 0.01 at point 22, ahead on 15 on the label-free angle (16 with labels runs). The key is results/arcs_rawchord/*/…gaps.manifold_minus_linear_raw.nearest_real_R.mean. The +0.011 figure is the headline arc only.
2. **Propagation.** The point-12 edits that are undone within four blocks are the smoothing-spline, chord and probe-QR edits. The interpolating spline on the label-free order stays 62° from true at point 16. Source: results/session2_propagation.json → readout_a.12.spline.err_to_true at read point 16.
3. **Scalar sagitta.** The p2-ring notes say 0.03–0.24 of centroid noise. That range is read from results/p2_steer_{speed,acceleration}_*_contiguous.json → sagitta_per_target[*].sagitta_over_centroid_noise at the 80/20 split: 0.031–0.088 and 0.033–0.092 for acceleration, 0.070–0.235 and 0.053–0.160 for speed. It does not reproduce the "0.04–0.19" in REPORT §3.4 from those keys.

## p2-shapes velocity sheet (verified 15:10 ET from results/p5_velocity_sheet.json, layers.22.steer)
- direction steer to a held-out (speed, direction) cell: sheet 2.881° vs ring1d 6.900° vs chord_raw 4.385° (direction.summary.*.endpoint_err); speed leakage sheet 0.034 vs ring 0.081 vs chord 0.051 m/s (offtarget_change)
- speed steer: sheet 0.108 vs speedline 0.121 m/s (speed.summary.*.endpoint_err); direction leakage sheet 1.235° vs speedline 3.041° vs chord_raw 11.448° (offtarget_change)
- gap sheet − chord_raw on direction endpoint: −1.504 [−1.901, −1.106] (direction.gaps.sheet_minus_chord_raw.endpoint_err); sheet − ring1d −4.019 [−4.898, −3.131]
- joint steer: sheet 4.334° / 0.110 m/s vs sequential 8.685° / 0.141 (joint.summary)
- geometry pt 22: cone R² 0.827 > cylinder 0.789, TPS 0.834, noise ceiling 0.902; ring radius by speed bin 4.9 → 11.4 (radius_noise_corrected); random init cone 0.29 / ceiling 0.54
- point 12: sheet 4.208° vs ring1d 6.674° vs chord_raw 4.629°; speed leakage 0.050 vs 0.182 vs 0.044
| p1-zone | transfer 0.71 / 0.18 / 0.58 (blocks 1 / 9 / 12) | results/p1a_perpatch_hard_seeds.json by_point[point].cross_half_r2.mean (0.708, 0.181, 0.580) |
| p2-compare | forecast 17° / 27° / 44° (interpolating 11°) | results/session2_predictor_native_readout.json per_layer.22.{probe_qr,chord,spline_smooth,spline}.dir_err_to_target.mean (17.13, 27.25, 44.19, 11.32) |
| p2-readuse | speed 0.31 m/s; spline − chord −0.005 [−0.007, −0.004] | results/session3_speed_predictor.json predictor.22.own.spline.direct_err_to_target (0.312); paired.22.own.spline_minus_chord_direct_err |
| interpret | disk tokens 0.88 → 0.22, background 0.12 → 0.77 (blocks 12 → 22); token shares 0.13 / 0.87 | results/p5_token_patching.json pair_types.random.readers.forecast.points.{12,22}.{obj,bg}.binding_fraction.mean (0.880, 0.218, 0.120, 0.770); config.plan.random.token_frac_mean.obj 0.1336 |
