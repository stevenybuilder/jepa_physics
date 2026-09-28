# Visible numbers in slides_v2

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
| p2-methods | 4.7°, 9.7° | REPORT.md §1 (headline arc, point 12: raw-centroid chord 4.7°, smoothing spline 9.7°) |
| p2-methods | conceptor, geodesic | pending: results/arcs_conceptor/*/p2_conceptor_direction_L12.json; geodesic file name to be confirmed |
| p2-readuse | 0.976 | results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 14 (0.9763) |
| p2-readuse | 0.939 | same file → curves.perpos_mean_r2 at point 24 (0.9390) |
| p2-readuse | 11.3° | results/session2_predictor_native_readout.json → per_layer.22.spline.dir_err_to_target.mean (11.3174) |
| p2-shapes | 4 values + title slot | pending: results/p5_velocity_sheet.json, p2_sheet_speed_L12.json, p5_time_manifold.json, p5_relational_motion.json |
| p1-binding | 2 values + title slot | pending: results/p5_object_vs_scene_direction.json, p5_relational_motion.json |
| p2-repair | 3 values + title slot | pending: results/p5_repair_attribution_L12.json (two rows); saddle-axis file name to be confirmed |
| p2-generalise | 2 values + title slot | pending: results/p2_steer_direction_direction_L12_contiguous_ctx-hard.json, p5_object_vs_scene_direction.json |

## Corrections applied in the notes

1. **Nearest-real agreement, spline minus raw chord.** Over the 16 arcs this is −0.02 ± 0.06 at point 12, with the spline ahead on 7, and +0.02 ± 0.01 at point 22, ahead on 15 on the label-free angle (16 with labels runs). The key is results/arcs_rawchord/*/…gaps.manifold_minus_linear_raw.nearest_real_R.mean. The +0.011 figure is the headline arc only.
2. **Propagation.** The point-12 edits that are undone within four blocks are the smoothing-spline, chord and probe-QR edits. The interpolating spline on the label-free order stays 62° from true at point 16. Source: results/session2_propagation.json → readout_a.12.spline.err_to_true at read point 16.
3. **Scalar sagitta.** The p2-ring notes say 0.03–0.24 of centroid noise. That range is read from results/p2_steer_{speed,acceleration}_*_contiguous.json → sagitta_per_target[*].sagitta_over_centroid_noise at the 80/20 split: 0.031–0.088 and 0.033–0.092 for acceleration, 0.070–0.235 and 0.053–0.160 for speed. It does not reproduce the "0.04–0.19" in REPORT §3.4 from those keys.
