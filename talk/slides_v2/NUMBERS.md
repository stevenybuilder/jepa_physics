# Visible numbers in slides_v2

The cites slide shows paper years only, from the "Slide citations" section of our reading list (not included).

Every number shown on a slide, with the file and key it was read from. Numbers inside figures are drawn from the same files by `talk/make_talk_figs_v2.py`. Speaker-note numbers are sourced in each `<aside>`.

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
| p2-compare (SUPERSEDED 18:xx, see v57 section) | 4.7° (chord) | same file → arms.chord_raw.err_probe_unmatched (4.7116); matches REPORT §1 |
| p2-compare (SUPERSEDED) | 9.7° (spline) | same file → arms.spline.err_probe_unmatched (9.7312); matches REPORT §1 |
| p2-compare | 0.16 m/s | results/p1c_direction_L9.json → single[n=5].off_target.mean_abs_change (0.160) |
| p2-compare | conceptor endpoint 165°, radius 0.70, speed 15 m/s | results/p2_conceptor_direction_L12.json arms.coast_a.{endpoint_err_deg.mean 165.03, probe_radius_min 0.699, off_target.abs_change_speed_mps.mean 15.24}; chord/spline radius arms.{linear_raw,manifold}.probe_radius_min 0.629 / 0.862 |
| p2-compare | off-target speed 0.12 / 0.43 / 0.19 m/s | results/p2_offtarget_direction_L12.json arms.{probe_qr,chord_raw,spline}.speed.mean 0.121 / 0.431 / 0.187; natural_spread.speed_mps 0.140 |
| p2-compare | notes: block-22 conceptor 41.8° [38.9, 44.5]; AND invalid 96% | results/p2_conceptor_direction_L22.json arms.coast_a.endpoint_err_deg; L12 aperture.pinv_AND_invalid_frac_by_alpha["0.1"] 0.964 |
| vs-paper (SUPERSEDED; no visible numbers now) | 4.7° vs 9.7°, 3.6° vs 10.7°; chord lead 1.3 / 2.3 over 16 arcs; 7/8 vs 0/8 stop-rule cells; 18 probes | same sources as p2-compare, p2-radius, p1-anisotropy and REPORT §3.2 (results/p1b_sameclip_direction_vs_speed.json) |
| p2-clock | 1.00 / 1.00 / 1.00 per step (slow / mid / fast); distance code 0.38 / 1.00 / 1.63 | results/p5_time_manifold.json → clock["direction/vjepa2/timepool"]["12"].spline.{slow,mid,fast}.slope.mean (1.003, 0.998, 0.996) and .odometer_prediction (0.377, 0.997, 1.626). The direction-set untrained values 0.51 / 0.58 / 0.94 (clock["direction/random/timepool"]["12"].spline.*.slope.mean 0.507, 0.575, 0.937) now appear only in the figure and notes; the rail shows the speed-set row below. Corrected 16:55 ET: previously misattributed to results/p5_clock_test.json (see results/CLOCK_NOTES.md) |
| p2-shapes (SUPERSEDED, v1; ledger #301) | sheet 2.9° vs 1-D ring 6.9° at point 22; speed leakage 0.03 m/s | results/p5_velocity_sheet.json |
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

## p2-table (added 16:55 ET; every value re-read from the raw JSON)

| Slide | Number on slide | File → key |
|---|---|---|
| p2-table | direction block 22: 10.69° / 7.45° / 7.98° / 12.58° | results/p2_bakeoff_direction_direction_L22_contiguous_rawchord.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (10.6897, 7.4453, 7.9775, 12.5803) |
| p2-table | direction block 12: 9.73° / 9.41° / 6.38° / 5.62° | results/p2_bakeoff_direction_direction_L12_contiguous_rawchord.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (9.7312, 9.4130, 6.3763, 5.6146) |
| p2-table | speed block 19: 0.092 / 0.088 / 0.132 / 0.526 m/s | results/p2_bakeoff_speed_speed_L19_contiguous.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (0.0920, 0.0880, 0.1324, 0.5262); the _storedbasis file gives probe_qr 0.2344 |
| p2-table | speed block 12: 0.163 / 0.155 / 0.162 / 0.433 m/s | results/p2_bakeoff_speed_speed_L12_contiguous.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (0.1629, 0.1548, 0.1617, 0.4332) |
| p2-table | acceleration block 21: 0.282 / 0.281 / 0.432 / 0.637 m/s² | results/p2_bakeoff_acceleration_acceleration_L21_contiguous.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (0.2819, 0.2814, 0.4323, 0.6374); the _storedbasis file gives probe_qr 0.6479 |
| p2-table | acceleration block 12: 0.309 / 0.305 / 0.527 / 1.537 m/s² | results/p2_bakeoff_acceleration_acceleration_L12_contiguous.json → arms.{spline,chord,chord_raw,probe_qr}.err_probe_matched (0.3087, 0.3049, 0.5266, 1.5371) |
| p2-table | direction forecast 11.3° vs 92.1° unedited (block 22, interpolating spline) | results/session2_predictor_native_readout.json → per_layer.22.spline.dir_err_to_target.mean (11.3174); per_layer.22.zero.dir_err_to_target.mean (92.1359) |
| p2-table | speed forecast R 0.62 at block 22; spline = chord | results/session3_speed_predictor.json → predictor.22.own.{spline,chord}.direct_R_speed (0.6223, 0.6239); "disk does not speed up" = REPORT §4.5 (displacement_R_speed 0.05) |
| p2-table | velocity sheet 2.9° vs 6.9° (1-D ring); speed leak 0.03 m/s | results/p5_velocity_sheet.json → layers.22.steer.direction.summary.{sheet,ring1d}.endpoint_err (2.8812, 6.9003); .sheet.offtarget_change (0.0339) |
| p2-table | time step: spline 0.106, raw chord 0.082, unedited 2.43 steps | results/p5_time_manifold.json → steer.22.summary.{spline,chord_raw,unedited}.time_err (0.1058, 0.0817, 2.4259); speed set |
| p2-table | untrained copy decodes the step, R² 0.97 | results/p5_time_manifold_controls.json → ["speed/random/timepool"].decode.22.test.t_r2 (0.9735; V-JEPA 2 same 768-clip subset 0.9980) |
| p2-table | "curve" for the time step | results/p5_time_manifold.json → geometry["speed/vjepa2/timepool"]["22"].heldout_var_explained.{spline,line_in_t} (0.823, 0.151) |
| p2-table | time-step code 1.00 / 1.00 / 1.00; distance code 0.38 / 1.00 / 1.63 | same keys as the p2-clock row (block 12, direction set) |
| p2-table | untrained flat at 0.65–0.77 | results/p5_time_manifold_controls.json → ["speed/random/timepool"].clock.{8,12,19,22}.spline.{slow,mid,fast}.slope.mean (min 0.653 at 8 fast, max 0.766 at 22 fast) |
| p2-table | relative speed, linear |v1 − v2| R² 0.945–0.976 (blocks 13–22) vs 0.19–0.22 untrained | results/p5_relational_motion.json → beyond_composition.sources.{vjepa2,random}[point 13..22].targets.rel_speed.lin_acts.test_r2 (V-JEPA 2 0.945 at 13 … 0.976 at 17/20; random 0.193–0.219); cross-check headline.best_by_target.abs_v_rel.{vjepa2_peak_oof_r2 0.969 at point 17, random_best_oof_r2 0.204}; untrained compositional null ≈ 0.92 = sources.random[p].targets.rel_speed.mlp_null.test_r2 (0.91–0.94, points 2–25). Replaces the earlier "nonlinear 0.94–0.98 vs 0.52" row (QA #280) |
| p2-table | disk tokens 0.88 → 0.22 of the forecast move | results/p5_token_patching.json → pair_types.random.readers.forecast.points.{12,22}.obj.binding_fraction.mean (0.8801, 0.2183) |

## vs-paper / interpret / p1-zone notes (ledger #272–#274, 16:55 ET)
- half-frame transfer 0.71 (block 1), −1.43 (block 8), 0.58 (block 12), 0.76 (block 22): results/p1a_perpatch_hard_seeds.json → by_point[point].cross_half_r2.mean (0.708, −1.432, 0.580, 0.761); vs-paper visible line and interpret claim 1 now cite token patching (row above: 0.88 → 0.22)

## QA fire #26 (#279, #280, #282), added 17:30 ET

| Slide | Number on slide | File → key |
|---|---|---|
| p2-table | probe-QR at its own norm 3.49° / 4.25° / 0.053 / 0.085 / 0.162 / 0.336 | results/p2_bakeoff_{direction_direction_L22_contiguous_rawchord, direction_direction_L12_contiguous_rawchord, speed_speed_L19_contiguous, speed_speed_L12_contiguous, acceleration_acceleration_L21_contiguous, acceleration_acceleration_L12_contiguous}.json → arms.probe_qr.err_probe_unmatched (3.4898, 4.2466, 0.0525, 0.0849, 0.1616, 0.3358). "unmatched" = each arm at its own norm (the file's norm_matching string) |
| p2-table (blue = own-norm winner, 5 of 6 rows) | own-norm comparators | same files → arms.{spline,chord,chord_raw}.err_probe_unmatched; next best: L22 chord_raw 3.6418, L12 chord_raw 4.7116, speed L19 chord_raw 0.0704, speed L12 chord_raw 0.1042, acc L21 chord 0.2663; acc L12 chord 0.3051 < probe_qr 0.3358 (not blue) |
| p2-table notes | MLP reader: direction 19–33° at the spline's norm, 17–33° at own norms; speed 0.39–0.60 m/s; unedited 86–92° | same files → arms.*.err_mlp_matched (direction 19.30–32.61), arms.*.err_mlp_unmatched (17.22–33.38), speed arms.*.err_mlp_{matched,unmatched} (0.392–0.601); unsteered_err.mlp (86.38, 92.30) |
| p2-clock | untrained, speed set 0.71 / 0.74 / 0.77 | results/p5_time_manifold_controls.json → ["speed/random/timepool"].clock.22.spline.{slow,mid,fast}.slope.mean (0.713, 0.743, 0.766) |
| p2-clock | shared curve 0.81 vs 0.01 | results/p5_time_manifold_controls.json → ["speed/vjepa2/timepool_same_subset"].geometry.22.heldout_var_explained.spline (0.813) vs ["speed/random/timepool"].geometry.22.heldout_var_explained.spline (0.009); the full speed set gives 0.823 (p5_time_manifold.json geometry["speed/vjepa2/timepool"]["22"]) |
| p2-clock / interpret notes | untrained fast/slow ratio 0.90–1.07 (blocks 8–22); unit slope 0.99–1.00 vs 0.65–0.77 | results/p5_time_manifold_controls.json → ["speed/random/timepool"].clock.{8,12,19,22}.spline.fast.slope.mean / .slow.slope.mean (0.984, 0.902, 0.962, 1.074); ["speed/vjepa2/timepool_same_subset"].clock.{8..22}.spline.*.slope.mean (0.987–1.002) |
| p2-clock / p2-table notes | time edit leaves decoded position unchanged: 0.479 vs 0.477 m | results/p5_time_manifold.json → steer.12.summary.{spline,unedited}.xy_err (0.4794, 0.4774); donor_real_step 0.1935; block 22: 0.4532 vs 0.4511 |

## p2-clock figure regenerated on the speed set (17:45 ET): figures/fig_p2clock_speedset.png, rail aligned to it

| Slide | Number on slide | File → key |
|---|---|---|
| (SUPERSEDED) p2-clock (figure + rail) | V-JEPA 2 0.99 / 1.00 / 1.00 per step, CI [0.983, 0.994] / [0.993, 0.998] / [0.994, 0.998] | results/p5_time_manifold_controls.json → ["speed/vjepa2/timepool_same_subset"].clock.22.spline.{slow,mid,fast}.slope.{mean,ci95} (0.9885, 0.9959, 0.9960); n_clips 100 / 135 / 63 |
| (SUPERSEDED) p2-clock (figure + rail) | untrained copy 0.71 / 0.74 / 0.77, CI [0.63, 0.80] / [0.69, 0.80] / [0.71, 0.83] | same file → ["speed/random/timepool"].clock.22.spline.{slow,mid,fast}.slope.{mean,ci95} (0.7128, 0.7432, 0.7659) |
| (SUPERSEDED) p2-clock (figure + rail) | distance counter 0.21 / 1.01 / 1.59 (hollow ochre) | same file → ["speed/vjepa2/timepool_same_subset"].clock.22.spline.{slow,mid,fast}.odometer_prediction (0.2144, 1.0078, 1.5945); identical under speed/random/timepool |
| p2-clock | supersedes the rail values 1.00 / 1.00 / 1.00 and 0.38 / 1.00 / 1.63 (direction set, block 12, row above), which stay in the notes only | results/p5_time_manifold.json clock["direction/vjepa2/timepool"]["12"] |

## Deck v56 rewrite (Opus deck designer, 2026-09-28 ~17:40 ET)

Plain names on slides: "straight edit" = raw-centroid chord (chord_raw / linear_raw); "curved edit" / "the curve" = spline (manifold); "the paper's probes" = probe-QR multi-probe; "gate" = COAST conceptor; "energy path" = Eq. 6 density-metric geodesic.

| Slide | Number on slide | File → key |
|---|---|---|
| design | rungs 1–5 (labels only) | none (ladder labels) |
| methods | 64 main directions; 80/20 split; 26 depths | PCA-64 and split per REPORT §2 / setup notes; 26 = embedding + blocks 1–24 + final norm |
| prior | blocks 8–9 | refs/physics_paper.txt (the paper's emergence zone; Table 1 l.85–95; ledger #285) |
| found, p2-readuse, p2-table | 11° vs 92° / 11.3° / 92.1° | results/session2_predictor_native_readout.json → per_layer.22.spline.dir_err_to_target.mean (11.3174); per_layer.22.zero.dir_err_to_target.mean (92.1359) |
| found, p2-binding | 0.88 → 0.22; 13% of tokens | results/p5_token_patching.json → pair_types.random.readers.forecast.points.{12,22}.obj.binding_fraction.mean (0.8801, 0.2183); config.plan.random.token_frac_mean.obj (0.1336) |
| found, p2-radius, p2-compare | within 1° (curve vs straight edit at the endpoint) | results/p2_endpoint_diagnosis.json → decomposition.L22_labels.arcs16.b_smoothing.interp_causalab_lam0.value (+0.837, straight better on 12 of 16: n_arm_better 4); decomposition.L12_labels.arcs16.b_smoothing.interp_causalab_lam0.value (+1.559; n_arm_better 2) |
| found | first spline trailed by 5–7° | results/p2_endpoint_diagnosis.json → decomposition.L12_unsup.headline_seed0.baseline_gap_ours_smooth_minus_raw_chord.value (5.02); decomposition.L22_labels.headline_seed0.baseline_gap_ours_smooth_minus_raw_chord.value (7.048) |
| found notes, p2-compare notes | authors' code on our knots, headline arc block 22: −0.48°; CV smoother −0.45° / −0.85°, 16 of 16 (exploratory) | results/p2_endpoint_diagnosis.json → decomposition.L22_labels.headline_seed0.b_smoothing.interp_causalab_lam0.value (−0.48); decomposition.{L22,L12}_labels.arcs16.b_smoothing.reinsch_lam_cv.{value,n_arm_better} (−0.447, 16; −0.847, 16) |
| (SUPERSEDED by v2 rows below) | 4.3° vs 8.7° | results/p5_velocity_sheet.json → layers.22.steer.joint.summary.sheet.endpoint_err_dir (4.334); joint.summary.seq_dir_then_speed.endpoint_err_dir (8.685) |
| p2-binding | 0.88 (big number) | as the found row above |
| (SUPERSEDED) p2-binding notes | per-patch R² 0.975 at block 12 | results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 12 (0.975) |
| p2-ring | −0.98 | results/p2_geometry_direction_L12.json → angle.circular_corr (−0.9832) (unchanged row) |
| p2-radius | 0.23 (big number, steel blue) | results/p2_steer_direction_direction_L12_contiguous_rawchord.json → min(waypoint_readout.linear_raw[3].radius) (0.2305) (unchanged row) |
| p2-readuse | 11.3° (big number); 92.1° unedited | as the found row above |
| p2-compare | paper's probes 3.5° | results/p2_bakeoff_direction_direction_L22_contiguous_rawchord.json → arms.probe_qr.err_probe_unmatched (3.4898; headline arc, own norm) |
| p2-compare (SUPERSEDED: now read from p2_coast_faithful_L22.json, see v57) | straight edit 4.2° (16 arcs, block 22) | results/p2_conceptor_direction_16arc_L22.json → arms.linear_raw.endpoint_err_deg.mean (4.2); block 12: results/p2_conceptor_direction_16arc_L12.json same key (5.7) |
| p2-compare | gate 97° (16 arcs, block 22) | results/p2_conceptor_direction_16arc_L22.json → arms.coast_a_b0.3.endpoint_err_deg.mean (96.9); L12 94.2; best variant arms.coast_b_dose_matched 75.7 / 83.7; curve (first smoother) arms.manifold 6.6 / 10.5 |
| p2-compare | heading left along the way: straight 0.63, curve 0.86 | results/p2_conceptor_direction_L12.json → arms.{linear_raw,manifold}.probe_radius_min (0.6289, 0.8616) (unchanged row) |
| p2-compare | forecast error, block-22 edit: 17° / 27° / 11° | results/session2_predictor_native_readout.json → per_layer.22.{probe_qr,chord,spline}.dir_err_to_target.mean (17.13, 27.25, 11.32); the curved-edit cell is the interpolating spline (smoothing spline 44.19 in notes) |
| p2-compare notes | energy path min radius 0.77 (kNN) / 0.69 (KDE) vs chord 0.70, spline 0.90, block 12, 3 targets, LITE, UNCONVERGED | results/p2_geodesic_direction_L12.json → headline.pooled_arms_mean.{geo_knn_from_chord,geo_kde_from_chord,chord,spline}.probe_radius_min (0.7701, 0.6948, 0.7006, 0.8952); convergence 1 of 42 solves (results/GEODESIC_NOTES.md) |
| p1-training | 0.96 / 0.38 | unchanged rows (curves.perpos_mean_r2 point 6; random max) |
| p1-zone | fails (blocks 4–8); works again from block 9 | results/p1a_perpatch_hard_seeds.json → by_point[4,6,7,8].cross_half_r2.mean (all < 0); by_point[9].cross_half_r2.mean (+0.181) (unchanged rows) |
| p1-anisotropy | 37; 3.2° | unchanged rows |
| p1-anisotropy notes | Makelov ranking: ρ −0.60 (p 3e-4), −0.77 by round; first eight 31°; 17–64 plateau 12–14°; random 13.4°; raw chord 34.7° | results/p5_makelov_ranking_L22.json → spearman_inlp_order_vs_move.{angle_move,by_round_angle_move}.rho; controls.{inlp_mean_move_deg,random_mean_move_deg,rawchord_move_deg}; per_direction (values as verified by the lead; REPORT §4.4) |
| p2-repair | 9% / 68% | results/p5_repair_attribution_L12.json (unchanged row) |
| p2-clock (not in main order) | 0.99–1.00 / 0.71–0.77 | results/p5_time_manifold_controls.json → ["speed/vjepa2/timepool_same_subset"] / ["speed/random/timepool"].clock.22.spline.{slow,mid,fast}.slope.mean (unchanged rows, shown as ranges) |
| p2-table (not in main order) | 11° vs 92°; 4.3° vs 8.7° | as the rows above |
| p2-beyond | no visible numbers | notes cite results/p2_geometry_direction_L12.json, results/session3_speed_predictor.json, results/p2_bakeoff_acceleration_acceleration_L21_contiguous.json, results/p5_velocity_sheet.json, results/p5_token_patching.json, results/p5_relational_motion.json, results/p5_time_manifold{,_controls}.json, results/p2_sheet_speed_L{12,19}.json |
| close | no visible numbers | notes reuse the rows above |
| p1-training | VideoMAE every patch, block 6: 0.82 | results/p1a_perpatch_direction_videomae.json → curves.perpos_mean_r2 at point 6 (0.817; points [0,1,2,4,5,6,8,9,12,16,22,24]) |
| p1-training notes | VideoMAE 0.961 vs V-JEPA 2 0.958 at block 8; onset 8 vs 5 | results/p1a_perpatch_direction_videomae.json → curves.perpos_mean_r2 at point 8 (0.961), onsets.perpos_mean_r2.onset (8, ci95 [8, 8]); results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 8 (0.958), onsets.perpos_mean_r2.onset (5, ci95 [5, 5]) |
| found, p2-readuse | 9% left (block-12 edit at block 24) | results/p5_repair_attribution_L12.json (same row as p2-repair above: 9% left at 24, edit at 12) |
| p2-ring (merged with p2-radius) | 0.23 (big number); "within 1°" in the takeaway | rows above (waypoint_readout.linear_raw[3].radius min 0.2305; p2_endpoint_diagnosis.json interp_causalab_lam0 arcs16) |

## Deck v57 (Opus deck engineer, 2026-09-28 ~18:15 ET). Every value re-read with a python one-liner from the named file

| Slide | Number on slide | File → key |
|---|---|---|
| p2-binding (visible line), found notes | untrained copy: disk 0.33 / 0.40 / 0.27, background 0.57 / 0.60 / 0.61 at blocks 8 / 12 / 22 | results/p5_token_patching_random.json → pair_types.random.readers.forecast.points.{8,12,22}.{obj,bg}.binding_fraction.mean (0.335, 0.400, 0.268; 0.571, 0.604, 0.612) |
| p2-binding notes | trained disk 0.98 / 0.88 / 0.22, background 0.04 / 0.12 / 0.77 | results/p5_token_patching.json → same key (0.979, 0.880, 0.218; 0.035, 0.120, 0.770) |
| p2-binding notes | forecast reader CV R² 0.35 untrained vs 0.91 trained | p5_token_patching_random.json → readers.forecast.cv_r2 (0.345); p5_token_patching.json → readers.forecast.cv_r2 (0.907) |
| p2-shapes (big number, caption), found | 4.3° vs 7.0°; 0.11 vs 0.16 m/s (notes) | results/p5_velocity_sheet_v2.json (mtime 17:36:53) → results.22.block2.joint.own.summary.{sheet,seq_global_interp}.{err_dir,err_spd}.mean (4.334, 6.980; 0.110, 0.161) |
| p2-shapes title "3–6°" | −2.6 [−3.2, −2.1] (block2), −5.5 (3×3 block), −6.2 (cross) | same file → results.22.block2.joint.own.gaps_sheet_minus.seq_global_interp.err_dir (−2.644, CI [−3.180, −2.142]); results.22.block3.joint.own.summary.{sheet,seq_global_interp}.err_dir.mean (4.98 − 10.46); results.22.cross.joint.own.summary same (4.32 − 10.51) |
| p2-shapes title "probe's floor" | 4.7° | same file → results.22.block2.reader_floor.real.summary.real_target_clips.err_dir.mean (4.721); err_spd 0.145 |
| p2-shapes notes | direction only 2.9 / 4.1 / 3.5; block 12 ring ties or wins (4.21 vs 3.87); speed leak 0.034 vs 0.088 | same file → results.{22,12}.block2.direction.own.summary.{sheet,ring1d_band_interp,ring1d_band_reinschcv}.{err_dir,leak_spd}.mean |
| p2-shapes notes | cylinder 5.5 vs 7.9; radius frozen +0.75 (matched norm) | same file → results.22.block2.joint.matched.summary.{abl_cylinder_from_sheet 5.538, seq_global 7.937, abl_radius_frozen 5.085, sheet 4.334} |
| p2-shapes notes, appendix | predictor 27.6 vs 31.5; twin 7.4; unedited 107 | results/p5_velocity_sheet_predictor.json → predictor.own.{sheet,seq1d_interp}.dir_err_deg.mean (27.61, 31.5); reference.{twin_forecast,unedited}.dir_err_deg.mean (7.4, 107.2) |
| p2-compare | 4.2° / 5.1° / 6.6° / 90° (16 arcs, block 22, each at its own norm); line "misses by 17°" at equal size | results/p2_coast_faithful_L22.json → arcs_full.arms.{chord_raw, spline_interp, spline, full\|contr\|pinv\|a0.5\|b0.1\|unc}.probe_own.mean (4.236, 5.073, 6.568, 90.21); spline_interp.probe_chordnorm.mean 17.27; unsteered_err.probe 88.7; norm ratio 16.92 / 11.85 from p2_endpoint_diagnosis configs.L22_labels.arcs16.arms.{add:spline_interp,add:chord_raw}.delta_norm |
| p2-compare line, p2-ring, p2-radius, vs-paper notes | authors' curve 0.8° behind at own norm (block 22); 1.6° at block 12; better on 4/16 and 2/16 | results/p2_endpoint_diagnosis.json → decomposition.{L22,L12}_labels.arcs16.b_smoothing.interp_causalab_lam0.{value,n_arm_better} (+0.837, 4; +1.559, 2) |
| p2-compare notes | our smoother gap 2.33 (block 22), 0.81 (block 12); headline 7.05 → −0.48 | same file → decomposition.L22_labels.arcs16.baseline_gap_ours_smooth_minus_raw_chord.value (2.332); configs.L12_labels.arcs16.arms.add:spline_smooth.probe_gap_vs_add_chord_raw.mean (0.81); decomposition.L22_labels.headline_seed0.{baseline_gap_ours_smooth_minus_raw_chord, b_smoothing.interp_causalab_lam0}.value (7.05, −0.48) |
| p2-compare notes | at the curve's norm the straight edit is 13.9° off | same file → configs.L22_labels.arcs16.arms.add_norm:chord_raw@spline_interp.probe.mean (13.89) |
| ledger #298 (notes only) | at the smoother's norm 6.64 vs 6.57 (tie) | same file → configs.L22_labels.arcs16.arms.{add_norm:chord_raw@spline_smooth, add:spline_smooth}.probe.mean |
| p2-compare, found | heading kept midway 0.88 (our smoother) vs 0.60 (straight), 16 of 16 arcs | results/arcs_rawchord/L22_s{1..16}[_labels]/p2_steer_direction_direction_L22_contiguous.json → summary.{manifold,linear_raw}.overall.probe_radius_min, mean over arcs (0.879, 0.600); block 12 0.845 vs 0.584 |
| p2-compare notes | energy path radius 0.65 vs chord 0.63, smoother 0.85 (block 22); block 12 0.71 / 0.63 / 0.88; 0 of 112 solves converged | results/p2_geodesic_direction_L{22,12}_full.json → headline.pooled_arms_mean.{geo_knn_from_chord,chord,spline}.probe_radius_min (0.653, 0.631, 0.847; 0.709, 0.629, 0.875); convergence.{batched_solves 56, converged 0} per file |
| (SUPERSEDED) p2-coordinates (big number), found | 0.76 vs 0.46 at block 12; winner at 6 of 7 blocks; untrained best x–y 0.34–0.53 | results/p5_motion_geometry.json → exp4_coordinate_search_and_exp5_shortcuts.points.12.per_coordinate.{fourier2_polar,cartesian}.encoding_r2_transfer_to_dirset (0.763, 0.456); winner_by_point (fourier2_polar at 1, 8, 12, 16, 19, 22; fourier_logpolar at 4); random_init.*.cartesian.encoding_r2_transfer_to_dirset (0.340–0.534, best at every point) |
| p2-coordinates line | straight 4-D edit ties the best curve: +0.13° [−0.03, 0.29], nearest-real R −0.001; vs raw chord −0.37°, +0.035 | same file → exp3_within_vs_between_and_fourier.fourier_dirset.12.differences.{fourier2_4d-spline_reinsch_cv_pca64, fourier2_4d-raw_chord}.{probe_err_deg,nearest_real_R}.mean |
| p2-coordinates line | speed and acceleration 19.5° apart; other pairs 79–89° | same file → exp1_whitened_metric.points.vjepa2.12.pairs.*.min_angle_deg |
| p2-readuse notes | random edit 2% left at block 24 vs 9% | results/p5_repair_attribution_L12.json → attribution_summary.{random12,spline12}.survival_final_block24.mean (0.020, 0.091) |
| p2-readuse notes | block-12 speed edit at natural norm R 0.57 / 0.70 | results/session3_speed_predictor.json → predictor.12.natural_norm.{spline,chord}.direct_R_speed (0.565, 0.703) |
| p1-zone | 0.18 at block 9; 0.58 at block 12; −1.43 at 8; 0.76 at 22 (notes) | results/p1a_perpatch_hard_seeds.json → by_point[point 9, 12, 8, 22].cross_half_r2.mean (0.181, 0.580, −1.432, 0.761) |
| p2-beyond, p2-clock notes, appendix | time edit +2 steps advances the forecast −0.21 [−0.42, −0.01] vs 1.15 real; reversal 0–7% vs 99% | results/p5_time_predictor.json → slide_numbers.{spline_+2_point22_advance_fraction, real_window_+2_advance_fraction, reversal} |
| p2-beyond notes | speed/acceleration lines | results/p5_speed_accel_angles.json → D_accel_vs_speed, A_rate_vs_displacement (values quoted without numbers on the slide) |
| p1-training notes | attentive probes add ≤ 0.015 | results/p1a_attentive_{direction,speed,acceleration}.json → layers[].attentive.cv_r2_mean vs layers[].meanpool_ridge.cv_mean (direction +0.015 at point 2, 0.987 vs 0.980 at 9, 0.995 vs 0.991 at 22) |
| design notes | 2,048 tokens (8 × 16 × 16), 26 read points | src/wm/extract.py → N_TOKENS, docstring (the 1,568 = 8 × 14 × 14 figure is VideoMAE at 224 px) |
| p2-generalise notes | plain-render probe reads the edits 67° off | unchanged, from the p2-generalise sources (not re-verified in v57) |
| p2-readuse (visible) | object's tokens alone 41° vs 80° frame-wide (block 12, natural dose) | results/session2_direction_natural_norm_disktokens.json → table.chord.err_to_target.mean (40.85, CI [35.6, 46.3]), R 0.618, obj_token_fraction.mean 0.091; results/session2_direction_natural_norm.json → table.12.natural.chord.err_to_target.mean (79.74, CI [72.5, 86.6]); unedited.err_to_target 92.14; twin_ceiling.err_to_target 9.11 |
| p2-readuse notes | survival at natural dose 17.5% (block 16), 11% (block 24) | session2_direction_natural_norm.json → survival.12.natural.chord.{pt16,pt24}.mean (0.175, 0.107) |
| p2-beyond notes | acceleration through the predictor (no numbers quoted) | results/session3_acceleration_predictor.json — lead-supplied summary, NOT re-verified in v57 |

## Round 3 (Opus deck engineer, ~18:45 ET). Every value re-read from the named file

| Slide | Number | File → key |
|---|---|---|
| p2-compare (figure fig_bakeoff_unified.png, notes), p2-table-appendix direction rows, found | block 22 / 12 own-norm endpoint: model-frame straight 3.68 / 4.75; paper's probes 4.24 / 4.12; raw chord 4.20 / 6.65; knot-CV curve (exploratory) 3.76 / 6.27; paper's spline 5.02 / 34.53; our smoother 6.46 / 8.13 (16 arcs each) | results/p2_bakeoff_unified_16arc.json → table.{fourier2_4d,probe_qr,chord_raw,causalab_lam_cv,interp_causalab_lam0,fitpack_smooth}.err_end.{22,12}.{mean,ci95,n_arcs} |
| p2-compare line "17° off"; notes 22.8 | paper's spline 17.04 and paper's probes 22.78 at the chord's norm, block 22 | same file → table.{interp_causalab_lam0,probe_qr}@chord_norm.err_end.22.mean |
| p2-compare notes | dose ratios 1.4× / 1.5× | same file → table.{interp_causalab_lam0,probe_qr,chord_raw}.delta_norm_end.22.mean (16.84, 17.76, 11.81) |
| p2-compare notes, found "0.92 vs 0.61" | path min radius: curves 0.86–0.94, straight edits 0.59–0.63; paper's spline 0.92 vs chord 0.61 at block 22 | same file → table.*.radius_min.{22,12}.mean (interp 0.915, chord 0.606) |
| p2-compare line "91–102°" | gate over 16 arcs at the chord's norm, all full and PCA-16 conceptor arms; unsteered 88.7–88.9 | results/p2_coast_faithful_L{12,22}.json → arcs.arms.{full\|…,pca16\|…}.probe_chordnorm.mean (L22 91.0–95.4, L12 92.2–102.0; gap_vs_chord_probe_chordnorm.n_arcs_ci_above_zero = 16 for every arm); unsteered_err.probe; REPORT §4.4 "COAST as written, over 16 arcs" |
| p2-readuse, p2-binding, found notes | disk tokens 40.8 (R 0.62); background count/dose 88.9 (R 0.09); background energy 80.1 (R 0.25); disk at 16 / 19: 67.0 / 70.6 | results/session2_direction_natural_norm_bgcontrol.json → table.{disk12,bg_count,bg_energy,disk16,disk19}.{err_to_target,R}.mean |
| p2-binding notes | encoder-output reader within 0.03 of the forecast reader | results/p5_token_patching.json → pair_types.random.readers.{forecast,encoder_output}.points.{0,8,12,16,22}.obj.binding_fraction.mean (max diff 0.027 at 12); disk 0.989, 0.979, 0.880, 0.761, 0.218 |
| limits notes | n: 64 + 64 pairs; 128 clips; 48 carriers; 200 × 4; 96 + 96 | p5_token_patching.json pair_types.{random,posmatch}…binding_fraction.n (64, 64); p5_time_predictor.json n_carriers_scored (128); p5_velocity_sheet_predictor.json n_carriers (48); session2_direction_natural_norm.json {n_carriers, n_targets} (200, 4); p5_time_static_control.json n_clips (96); p2_bakeoff_unified_16arc.json keys.points (16 arcs × 8 targets × 48 clips) |
| p2-beyond, p2-clock | static disk 0.55–1.07 of the moving rate; untrained 1.02–1.10; 2.3–2.9× farther | results/p5_time_static_control.json → verdict (and models.*.*.static_over_* per keys) |
| p2-contact (big number, caption, lines) | 0.80 after / 0.22 before (0.43 straddle); 86% closer to reflected; forecast 0.17 / 0.09 / 0.07 / 0.00 vs real 0.86 / 0.75 / 0.53 / 0.67 (n 12 / 29 / 60 / 60); 96 clips | results/p5_contact_dynamics.json → results.per_point.22.bounce.{pre_contact,straddle,post_contact}.turn_fraction.mean, post_contact.frac_closer_to_out (0.861); results.predictor_forecast.steps.step{0..3}_tubelet{4..7}.{forecast_turn_fraction_post.mean, real_future_encoder_turn_fraction_post_point25.mean, n_post_contact}; results.n_bounce (96) |
| p2-contact figure | turn fraction by frame offset −6…6 | same file → results.per_point.22.bounce.turn_fraction_by_offset.* (talk/make_talk_figs_v2.py contact()) |
| p2-compare figure whiskers, notes, limits notes | arc-level mean ± 2 SE over 16 arcs: block 22 half-widths 0.16 (model-frame), 0.71 (probe steer), 0.22 (raw chord), 0.20 (CV curve), 0.80 (paper's spline), 0.82 (our smoother); block 12 0.10–1.06 (paper's spline 21.7) | results/endpoint_diagnosis_raw/unified_L{22,12}_{a,b}.json → per-arc mean of arms.<arm>.err_end (8 arcs per file), mean ± 2·SD/√16 (talk/make_talk_figs_v2.py _arc_means) |
| p2-compare notes | probe-subspace steer at the chord's norm, block 12: +0.27 [0.03, 0.49] vs raw chord | results/p2_bakeoff_unified_16arc.json → paired_vs_chord["probe_qr@chord_norm - chord_raw"].err_end.12 |
| p1-anisotropy notes | nonlinear reader after erasing the rank-2 plane: 0.37 / 0.41 / 0.82 (blocks 9 / 12 / 22) | results/p1b_dims_four_ways.json → variables.direction.layers[point].leace.mlp_r2_after (0.374, 0.410, 0.823); control.*.leace (planted controls) |
| p1-overview (figure fig_part1_panels.png, notes) | block-1 R² 0.875 / 0.983 / 0.977, untrained 0.848 / 0.920 / 0.914; K 37 / 39 / 41; 8.7° at N = 5 | results/p1a_{direction_direction,speed_speed,acceleration_acceleration}_meanpool{,_random}.json → layers[1].cv_mean; p1b_{direction,speed,acceleration}_*_meanpool_L9.json → K; p1c_direction_L9.json → random_nulls.rows[n=5].learned.mae_to_target (8.73) |
| p2-variables notes | frame-count patch −0.17 [−0.32, −0.005] (block 22), 0.34 (block 12); later-slot tokens 0.84 / 1.56; pooled −0.21; real window 1.15 | results/p5_time_patch_predictor.json → arms.{22,12}.{tp+2,tok+2}.advance_fraction; pooled_push_same_clips.22.sp+2; references.win+2 |
| p2-variables notes | radial edit moves forecast disk 15–30 px; relative speed R² 0.969 (untrained 0.204); acceleration forecast 1.82 vs 3.46 unedited (twin 0.83) | results/p5_radial_steering_L22.json → radial.{r025,r05,r15,r2}.pos_shift_px.mean; p5_relational_motion.json → headline.best_by_target.abs_v_rel; session3_acceleration_predictor.json → predictor.21.own.spline.direct_err_to_target.mean, twin_reference.direct.{unedited_err_to_target,twin_forecast_err_to_target} |
| (SUPERSEDED) p2-shapes notes | sheet − ring-only through the predictor: +1.54 [−1.93, 5.13] own norm; +6.72 [2.82, 10.38] at the sheet's norm (n = 48) | results/p5_velocity_sheet_predictor.json → paired.{own,sheet_norm}.sheet_minus_ring1d_dir_err |

## Round 5 (~19:00 ET)

| Slide | Number | File → key |
|---|---|---|
| p2-coordinates (big number, caption, figure), found row 2 | polar − x–y at rank 2: +0.015 [0.009, 0.022] at block 12; positive from block 9 except 11 and 22; ties before 9; untrained −0.009 to −0.020 (CI excludes 0) at blocks 1–25; block 0 ≈ 0 | results/p5_coordinate_competition.json → encoding.{vjepa2,random}.<pt>.primary_speedset.per_candidate.polar2.{minus_cartesian,minus_cartesian_ci95}; winner_by_point |
| p2-coordinates notes | polar 4-D 0.407 vs random 4-D 0.283 (best of 20: 0.473), transfer, block 12; no frame reaches matched PCA ceiling | same file → encoding.vjepa2.12.transfer_speed_to_dirset.r2.{polar4,randfeat_4,randfeat_4_max_of_20}; per_candidate.*.minus_pca_k < 0 at every point |
| p2-coordinates line, notes | model-frame straight edit min path radius 0.61 / 0.63 | results/p2_bakeoff_unified_16arc.json → table.fourier2_4d.radius_min.{12,22}.mean |
| p2-compare line | model-frame edit at the chord's norm 6.33 vs chord 4.20 (block 22) | same file → table.fourier2_4d@chord_norm.err_end.22.mean; table.chord_raw.err_end.22.mean |
| p2-compare notes | equal-size path radius: curves 0.71–0.86, straight 0.59–0.63 | same file → table.{causalab_lam_cv,fitpack_smooth,interp_causalab_lam0}@chord_norm.radius_min.*; table.{chord_raw,fourier2_4d,probe_qr}.radius_min.* |
| p2-ring notes | own-norm curves 0.80–0.94 (paper's spline 0.92 / 0.80) | same file → table.*.radius_min.{22,12}.mean |
| p2-readuse line | disk tokens only: straight 41°, curve 100° (unedited 92) | results/session2_direction_natural_norm_disktokens.json → table.{chord,spline}.err_to_target.mean (40.85, 100.24); unedited_err_to_target |
| p2-readuse notes | disk-token edit at blocks 16 / 19: 67.0 / 70.6 | results/session2_direction_natural_norm_bgcontrol.json → table.{disk16,disk19}.err_to_target.mean |
| p2-shapes (title, lines) | predictor n = 200: sheet 29.8 vs two interp curves 34.5 (−4.78 [−6.24, −3.29]); ring alone 28.2, speed error 1.17 vs 0.28 m/s | results/p5_velocity_sheet_predictor_n200.json → predictor.own.{sheet,seq1d_interp,ring1d_interp}.{dir_err_deg,speed_err_mps}.mean; paired.own.sheet_minus_seq1d_interp_dir_err; paired.own.sheet_minus_ring1d_interp_dir_err (+1.59 [0.13, 3.28]) |
| p2-shapes line | sheet at the probe edit's norm 70.05 vs probe edit 7.26 | results/p5_velocity_sheet_v2.json → results.22.block2.joint.own.summary.{sheet_at_probe_edit_norm,probe_edit}.err_dir.mean |
| found row 3, cover, close | curve's forecast 12° vs 26° (block 22, natural norm, one arc) | results/session2_direction_natural_norm.json → table.22.natural.{spline,chord}.err_to_target.mean (12.3, 25.7) |
| headline (new) | 11.3° / 92.1° / 9.1°; 9% vs 2% | session2_predictor_native_readout.json → per_layer.22.{spline,zero}.dir_err_to_target.mean; session2_direction_natural_norm.json → twin_ceiling.err_to_target.mean (9.11); p5_repair_attribution_L12.json → attribution_summary.{spline12,random12}.survival_final_block24.mean |
| p2-variables notes | speed–acceleration axis angle 19.5° trained, 14.3° untrained (block 12, whitened) | results/p5_motion_geometry.json → exp1_whitened_metric.points.{vjepa2,random}.12.pairs["speed\|acc"].min_angle_deg |
| vs-paper notes | sawtooth absent: lag-1 autocorrelation 0.42–0.95; C.12 "< 0.5°" vs 11.9° | REPORT §3.2 (l.391–396) from results/p1b_* ; REPORT §2 (c), (d) |
| p2-clock rail, p2-table-appendix | untrained slope 0.85–0.87 (block 22), 0.79–0.81 (block 12); decode R² 0.98; geometry 0.82 vs 0.009 (full speed set) | results/p5_time_manifold_controls.json (mtime 17:38) → ["speed/random/timepool"].clock.{22,12}.spline.*.slope.mean; .decode.22.test.t_r2; geometry.22.heldout_var_explained.spline; ["speed/vjepa2/timepool_same_subset"].geometry.22 (0.823) |
| p2-variables (verdict "via speed", notes), vs-paper | acceleration on 240 decorrelated clips: partial R² 0.44 / 0.79 (blocks 12 / 22); after removing decoded speed sequence −0.013 to 0.003; MLP on activations 0.20–0.53 vs on decoded speeds 0.93–0.94; untrained ≈ 0 | results/p5_accel_decorrelated.json → models.{vjepa2,random}.points.{12,22}.{meanpool,timepool}.{ridge_a_partial_mean_speed_displacement,ridge_a_partial_decoded_speed_seq}; .mlp.{act_meanpool,act_timepool,speedseq_inset_nested}; design_correlations_with_a.mean_speed (0.0) |
| p2-variables, p2-clock, p2-beyond, p2-table-appendix notes | frame count positional: slot coefficient 0.956 / 0.918, nearer-slot 0.999 / 0.988 (blocks 12 / 22), content ≤ 0.006; static slope 0.75 / 0.62 vs untrained 0.96 / 0.98; block 0 undecodable | results/p5_time_shuffle.json → models.{vjepa2,random}.<pt>.shuffled_pooled.{coef_slot,coef_content,frac_nearer_slot}; .static.slope_decoded_on_slot |

## Round 6 (~19:40 ET)

| Slide | Number | File → key |
|---|---|---|
| p2-contact (line), p2-variables notes, figure | bounce inside the context: forecast turn 0.84 [0.75, 0.93]; early 1.00, late 0.66; straight twins −0.002; per step 0.89 / 0.90 / 0.77 / 0.82 | results/p5_contact_in_context.json → results.forecast.step_mean_turn_fraction.{bounce,straight_in}.{all,early_kb2-3,late_kb5-6}.mean; results.forecast.steps.*.bounce.all.turn_fraction.mean |
| p2-contact (line) | unseen bounce, reader refit to read turned forecasts: 0.04–0.09 (≈ 0.07); a reader fit on all clips 0.31–0.33 at steps 3–4 (notes) | results/p5_contact_probe_domain.json → results.a_reader_refits.transfer.out_of_window_k_b>=8.step{0..3}.bounce_turn_fraction.mean; +all_xfit same key |
| p2-coordinates notes | steering at the chord's norm on constant-velocity carriers: x–y minus polar 2-D −0.41 (block 12), −1.58 (block 22) | results/p5_coordinate_competition.json → steering_posthoc_velocity_carriers.{12,22}.cartesian_minus_polar2.chord_norm.mean |
| p2-shapes (line, notes) | ring alone vs sheet on the disk's path heading: sheet −7.13 own norm, −7.45 sheet norm (interp ring) | results/p5_velocity_sheet_predictor_n200.json → paired.{own,sheet_norm}.sheet_minus_ring1d_interp_disp_heading_err |
| p2-variables, vs-paper notes | acceleration after an out-of-set speed decoder: 0.32 / 0.46 (timepool, blocks 12 / 22) | results/p5_accel_decorrelated.json → models.vjepa2.points.{12,22}.timepool.ridge_a_partial_transfer_speed_seq |
| headline notes | 9% vs 2% survival: 16 carriers, one arc | results/p5_repair_attribution_L12.json → attribution_summary.{spline12,random12}.survival_final_block24 (n 16) |
| p2-readuse (line), found, limits notes | source-disk tokens only (76 tokens): chord 44.7 [38.8, 50.4], R 0.60; polar straight 37.0; twin-only tokens 91.5; label-ordered spline 68.1; blocks 8 / 16 / 19: 86.0 / 69.1 / 76.5 | results/session2_disk_token_sweep.json → table.{src_chord_1x,src_fourier4_1x,twinonly_chord_1x,src_spline_labels_1x,src_chord_L8_1x,src_chord_L16_1x,src_chord_L19_1x}.err_to_target.mean; .R.mean |

## Round 7 (~20:00 ET)

| Slide | Number | File → key |
|---|---|---|
| p2-contact (line), p2-variables (steered ●, notes) | injected bounce: forecast turn 0.48 (block 22, all tokens), 0.45 (block 12, disk tokens); random ≈ 0; wrong-heading −0.27; pasted twin tokens 0.82 | results/p5_contact_steer.json → results.points.22.conditions.a_pool_loo.forecast_turn_fraction_step_mean.all (0.478); points.12.conditions.b_disk_loo (0.449); c_rand_pool (−0.022); d_far_pool (−0.266); e_own_tokens (0.822) |
| p1-layers | 0.875 / 0.848; speed 0.98, acceleration 0.98 at block 1 | results/p1a_{direction_direction,speed_speed,acceleration_acceleration}_meanpool{,_random}.json → layers[1].cv_mean |
| p1-nullspace | 37 / 39 / 41; random 0.98 | results/p1b_*_meanpool_L9.json → K; p1b_direction_direction_meanpool_L9.json → random.rows[].cv_r2 |
| (SUPERSEDED) p1-steer | 3.6° at N = 8 vs random 9.2° (p05 4.1); 8.7° at N = 5 (random 12.8, p05 6.2) | results/p1c_direction_L9.json → random_nulls.rows[n].{learned.mae_to_target, random_basis.mae_to_target.{mean,p05}} |
| p2-splines (figure fig_splines3.png) | held-out values and blocks | results/p2_steer_{speed_speed_L19,acceleration_acceleration_L21,direction_direction_L22}_contiguous.json → held_out_values; curves recomputed by talk/make_talk_figs_v2.py splines3 from artifacts/activations via wm.p2_data.load_inputs (knot clips, PCA-64, smoothing spline) |
| p1-steer (title, big number, line, figure) | ridge 8.73° at N = 5 vs same-rank random median 54.9 (p 0.035, 200 draws); Adam 69.2° at N = 5 (median 69.9, p 0.49), first ≤ 10° at N = 18 (9.32); split-half judge 14.7 at N = 5 | results/p1c_direction_L9_nulls200.json → bases.{ridge,adam}.table[n].{learned_mae_to_target, rank_matched.{median,p}}; evalprobe_variants.split_half.all_n[n=5] |
| p1-nullspace notes | speed and acceleration stop on the MAE rule at R² 0.20 / 0.16 | results/p1b_{speed_speed,acceleration_acceleration}_meanpool_L9.json → rounds[K].cv_r2 (0.197, 0.160; K 39, 41) |
| p2-coordinates notes | polar-frame straight edit through the predictor: 28.6 vs chord 25.7 vs spline 12.3 (block 22, natural norm) | results/session2_fourier4_predictor.json → task1.table.P22.natural.{fourier,chord,spline}.err_to_target.mean |
| p2-variables relative speed (shape ◐, steered ●) | ordered but curved; edit leaves v1, v2 unchanged | results/p5_relational_shape.json (ordered but curved; residual 0.39–0.44) |

## Round 8 (~20:45 ET)

| Slide | Number | File → key |
|---|---|---|
| headline notes, limits notes | probe-free: every block-22 edit moves the whole forecast 0.15–0.26 of the way to the twin's; natural norm spline 0.186 vs chord 0.234 (recovery) | results/session2_fourier4_predictor.json → task1.table.P22.{own,natural}.{fourier,spline,chord}.recovery_full.mean |
| p2-compare (line, notes) | probe steer at the chord's norm, block 22: 22.8° [22.0, 23.7], R 0.00 | results/p2_bakeoff_unified_16arc.json → table.probe_qr@chord_norm.{err_end,R_end_test}.22 |
| p1-reproduce (figure fig_p1_reproduce.png = fig_part1_panels.png, notes) | as p1-layers / p1-nullspace / p1-steer rows | same files; steering panel now same-rank median (p1c_direction_L9_nulls200.json bases.ridge.table[].rank_matched.median) and Adam basis (bases.adam.table) |
| p1-reproduce notes, p2-variables notes | acceleration magnitude on decorrelated clips: the **linear |a| ridge** reads ≤0.07 at blocks 1–9 (≤0.01 mean-pooled, ≤0.01 time-ordered); calibrated |signed prediction| reads it at the zone, 0.215 at block 8, 0.442 at block 9 (time-ordered features); ≤0.07 holds for the linear ridge only | results/p5_accel_decorrelated_magnitude.json → models.vjepa2.points.{1,4,8,9}.{meanpool,timepool}.abs_a.ridge.r2; points.{8,9}.timepool.abs_a_from_abs_signed_ridge.r2 (0.215, 0.442) |
| p2-contact (line) | seen 0.84; unseen 0.07 (native 0.17, at most 0.29); injected 0.48 | p5_contact_in_context.json; p5_contact_probe_domain.json a_reader_refits.{transfer,native,+all_xfit}; p5_contact_steer.json |
| p2-binding notes | disk-token edit 44.7 (twin dose) vs 65.8 (uniform dose) | results/session2_disk_token_sweep.json → table.{src_chord_1x,src_chord_uniform}.err_to_target.mean |
| p2-contact (title, line, notes), p2-variables (Contact steered "heading only") | heading edit without a wall 0.77 [0.68, 0.86] vs bounce edit 0.48 at block 22 (paired −0.29 [−0.38, −0.21]); block 12 disk tokens 0.46 vs 0.45 (paired −0.011 [−0.033, 0.009]) | results/p5_contact_steer_heading_control.json → results.points.{22,12}.forecast_turn_fraction_step_mean.{h_pool_loo,a_pool_loo,h_disk_loo,b_disk_loo}; results.points.*.paired |
| headline notes | own-norm probe-free: forced choice 0.045 vs 0.016, recovery 0.172 vs 0.154, twin id 0.331 vs 0.310 | results/session2_fourier4_predictor.json → task1.table.P22.own.{spline,chord}.{forced_choice_frac,recovery_full,twin_id_acc}.mean |
| limits notes | cached vs recomputed twin forecasts: median 1.6%, max 31%; ceiling 9.1° cached, 9.3° recomputed | ledger #375 (QA fire #34); not re-verified per key |
| p1-reproduce notes | 74-dim random basis at N = 5: median 11.8, p 0.22; beaten at N = 10, p 0.04 | results/p1c_direction_L9_nulls200.json → bases.ridge.table[n].rank_2K.{median,p} (ledger says first beaten at N = 9; the table has no N = 9 row) |
| p2-variables notes | signed acceleration 0.44 / 0.79; magnitude 0.12–0.16 at blocks 16–22 | p5_accel_decorrelated.json (partial R², verified); p5_accel_decorrelated_magnitude.json (lead's summary) |
| headline (line "Block-12 edit: 9% left; random 2%") | 9% vs 2% survival at block 24, separate 16-carrier repair run (smoothing spline, one target, one seed; chord 11%) | results/p5_repair_attribution_L12.json → attribution_summary.{spline12,random12,rawchord12}.survival_final_block24.mean (0.091, 0.020, 0.107) |
| headline notes | 11.3° own norm (12.3° natural); probe-free recovery 0.17, null 0.08 | results/session2_predictor_native_readout.json per_layer.22.spline; session2_fourier4_predictor.json task1.table.P22.{own,natural}.{spline,null}.{err_to_target,recovery_full} |
| p2-shapes notes | heading probe at the sheet's norm: ring 21.9 vs sheet 29.8 (paired +7.89 [6.12, 9.68]) | results/p5_velocity_sheet_predictor_n200.json → paired.sheet_norm.sheet_minus_ring1d_dir_err |
| p2-variables (cell "not as time", notes) | block-12 slot-code patch +0.34 [0.21, 0.49]; reversed slots at block 22 −0.40 | results/p5_time_patch_predictor.json → arms.12.tp+2.advance_fraction; arms.22.tp_rev.advance_fraction |
| p2-binding notes | no-probe token-swap shares 0.77 / 0.23; disks alone 4% carry 0.82 | from the QA fire #34 raw re-derivation (not re-read per key in this pass) |
| p2-compare notes | paper's spline 34.5° at block 12 uses label-free knot ordering (label-ordered 7.2° vs chord 5.7°) | results/p2_bakeoff_unified_16arc.json (keys.points: 12 = unsupervised angle) ; results/p2_endpoint_diagnosis.json configs.L12_labels.arcs16.arms.{add:causalab_interp,add:chord_raw}.probe.mean (7.23, 5.67) |
| p2-splines (figure fig_splines3.png, regenerated) | PCA plane fit on kept knot clips only | talk/make_talk_figs_v2.py splines3 |

## Round 9 (QA Part 1 fire #32, ledger #382–#389)

| Slide | Number | Source (file → key) |
|---|---|---|
| p1-reproduce notes; p2-variables notes | decorrelated |a| through the signed code: 0.22 at block 8, 0.44 at block 9 (time-ordered features); ≤0.07 at blocks 1–4 | results/p5_accel_decorrelated_magnitude.json → models.vjepa2.points.{8,9}.timepool.abs_a_from_abs_signed_ridge.r2 (0.215, 0.442) |
| p1-reproduce notes; p1-steer title | full 74-dim random null first beaten (p < 0.05, sustained) at nine ridge probes, p 0.040; at five p 0.22, median 11.8° | results/p1c_direction_L9_nulls200.json → bases.ridge.first_n.rank_2K.p_lt_0.05.{first_n,first_n_sustained} = 9; bases.ridge.all_n[n=9].rank_2K.p = 0.0398; [n=5].rank_2K.{p,median} (supersedes "ten beat it", which read the 12-row table) |
| p1-zone notes | hard render, largest per-position rise at blocks 4→6 on all three seeds: +0.29 / +0.25 / +0.24; 8→9 is +0.09 / +0.07 / +0.07 | results/p1a_perpatch_hard_seeds.json → by_point[point].perpos_mean_r2.per_seed, differences on common_points (0.286, 0.245, 0.237) |
| papers notes | "direction becomes accessible only at the Physics Emergence Zone"; probes trained with Adam | refs/physics_paper.txt l.47, l.121; l.1206 |
| p1-reproduce figure | label "(mean speed on this set)" on the acceleration curve | talk/make_talk_figs_v2.py part1_panels → talk/figures_v2/fig_p1_reproduce.png (= fig_part1_panels.png) |

## Round 9 add-on (narrative frame; twin-free dose landed)

| Slide | Number | Source (file → key) |
|---|---|---|
| p2-binding notes | twin-free dose profile 49.6° (probe-split clips) / 49.5° (all probe clips) vs per-token 44.7°, uniform 65.8°; recovers 77% [0.70, 0.82] of the uniform-to-per-token gap; residual over per-token +4.9° [3.7, 6.3] paired, n = 200 | results/session2_disk_token_sweep_twinfree.json → table.src_chord_twinfree_bin.err_to_target.mean (49.55); table.src_chord_twinfree_all.err_to_target.mean (49.49); frac_of_uniform_to_pertoken_gap_recovered.bin.{ratio,ci95} (0.768); paired.twinfree_bin_minus_pertoken.{mean,ci95} (4.90) |
| p2-compare (visible line, notes) | minimum path radius at block 22, 16 arcs: chord 0.61; curves 0.88 (our smoother), 0.92 (paper's spline), 0.94 (cross-validated); straight arms 0.60–0.63 | results/p2_bakeoff_unified_16arc.json → table.{chord_raw,fitpack_smooth,interp_causalab_lam0,causalab_lam_cv,fourier2_4d,probe_qr}.radius_min.22.mean. The lead's "0.63 vs 0.86" is the single headline arc (headline_arc.12.chord_raw 0.632, fitpack_smooth 0.865); the slide uses the 16-arc block-22 values |
| p2-compare notes | conceptor (COAST) gate overshoots to 165.0° at block 12 | REPORT.md §4.4 conceptor table, row "COAST gate, uncentred"; results/p2_coast_faithful_L12.json |
| p2-compare notes | energy geodesic dips into the hollow like the chord: min radius kNN / KDE 0.71 / 0.62 at block 12, 0.65 / 0.63 at 22, vs spline 0.88 / 0.85; 0 of 56 + 56 solves converged | results/p2_geodesic_direction_L{12,22}_full.json → headline.pooled_arms_mean.*.probe_radius_min; convergence (REPORT.md §4.4, headline arc only) |
| p2-compare, p1-zone notes | edits at blocks 2–12 wash out within four blocks (82–88° four blocks later) | REPORT.md §4.6 wash-out table |

## Round 10 (independent acceleration audit; notes only)

Audit = re-derivation from pixels and feature files (scratchpad qa_accel/s2_supplied_probes.json, s3_grid.json); "audit CI" = cell-block bootstrap over the grid's 20 cells, beside our clip-bootstrap CI.

| Slide | Number | Ours (file → key) | Audit |
|---|---|---|---|
| p1-reproduce notes | supplied set: acceleration and mean speed identical by construction (every clip starts at rest, mean speed = 0.3125 s × a); block-1 R² 0.977 → −0.006 with pixel-measured speed and displacement partialled out; constant-speed-set speed probe scores 0.975 on the acceleration clips | results/p1a_acceleration_acceleration_meanpool.json (0.977) | s2_supplied_probes.json → vjepa2_L1.{plain 0.9769 ± 0.0018, partial_measured_ms_disp −0.006 ± 0.008, speedset_probe_transfer 0.975} |
| p1-reproduce, p2-variables notes | signed a beyond mean speed and displacement, mean-pooled: block 1 −0.008, 4 −0.027 (≈0), 8 0.37 [0.27, 0.46], 9 0.27 [0.17, 0.38], 12 0.44 [0.33, 0.53], 22 0.79 [0.75, 0.83] | results/p5_accel_decorrelated.json → models.vjepa2.points.{1,4,8,9,12,22}.meanpool.ridge_a_partial_mean_speed_displacement.{r2,ci95} | s3_grid.json vjepa2_meanpool_L{p}.signed: 1 −0.00, 4 −0.04, 8 0.37 audit CI [0.10, 0.46], 9 0.29 audit CI [0.003, 0.39], 12 0.44 audit CI [0.17, 0.56], 22 0.79 audit CI [0.67, 0.84]. Notes quote our file (0.27 at block 9); the audit re-derivation, also partialled, gives 0.29 |
| p2-variables notes | per-tubelet (time-ordered) features read signed a from block 1: 0.53 [0.46, 0.60] | results/p5_accel_decorrelated.json → points.1.timepool.ridge_a_partial_mean_speed_displacement | s3_grid.json vjepa2_timepool_L1.signed 0.531, audit CI [0.34, 0.59] |
| p2-variables notes | |a| weak on the grid: ≤0.17 late (linear ridge, mean-pooled 0.164 at 22, 0.118 at 16); cell-block CI includes zero | results/p5_accel_decorrelated_magnitude.json → models.vjepa2.points.{16,22}.meanpool.abs_a.ridge.r2 | s3_grid.json vjepa2_meanpool_L22.abs 0.167, audit CI [−0.22, 0.35]; L16 0.116, audit CI [−0.22, 0.26] |
| p2-variables notes | grid |a| ≤ 3 m/s² vs the paper's a ∈ {2, 4, 6, 8, 10} m/s² | results/p5_accel_decorrelated_magnitude.json → abs_a_levels {0, 1.5, 3.0} | refs/physics_paper.txt l.683 |

## Round 11 (QA fire #36, Part 2)

| Slide | Number | Source (file → key) |
|---|---|---|
| headline (title, notes), close notes, found (row, notes) | a block-12 edit through the disk's tokens reaches the forecast's heading: per-token chord 44.7°, union 40.8°, Fourier at natural size 33.0°, vs 92.1° unedited | results/session2_disk_token_sweep.json → table.{src_chord_1x,union_chord_1x}.err_to_target.mean (44.66, 40.85); results/session2_fourier4_predictor.json → task1.table.D12.natural.fourier.err_to_target.mean (33.0); stored_references.unedited_err 92.1 |
| found (row "Probe ties; forecast wins", notes), p2-compare (subtitle, notes) | through the predictor at block 22 the spline lands 11.3° vs chord 27.2° at own size, 12.3° vs 25.7° at natural size | results/session2_fourier4_predictor.json → task1.table.P22.{own,natural}.{spline,chord}.err_to_target.mean |
| p2-compare notes | COAST gate (uncentred, pca64, α 0.1) at block 12: 114.4° at β = 0.3 (inside COAST's β ∈ {0.1, 0.3}); 165.0° at β = 1 | results/p2_coast_faithful_L12.json → arms["pca64\|contr\|jaeger\|a0.1\|b0.3\|unc"].probe_own (114.4), arms["pca64\|contr\|jaeger\|a0.1\|b1\|unc"].probe_own (165.0) |
| p2-splines notes | our smoother is the weakest direction arm at block 22 only; at block 12 the paper's spline 34.5° vs ours 8.1° | results/p2_bakeoff_unified_16arc.json → table.{interp_causalab_lam0,fitpack_smooth}.err.12 (figure fig_bakeoff_unified.png shows 34.5, 8.1) |
| p2-repair notes (backup) | natural-size block-12 edit survives 0.107 [0.100, 0.113] at block 24, n = 200 | results/session2_direction_natural_norm.json → survival.12.natural.chord.pt24 |
| limits notes | cache mismatch median 1.6% of the forecast's size ≈ 7% of a typical twin change; max ≈ 1.3 twin changes; ceiling 9.1° cached vs 9.26° recomputed | lead's QA #397 figures from REPORT §7 (not re-derived per key in this pass); supersedes the Round-QA#34 "1.6 percent of the twin change, at most 31" wording |
| p2-variables notes | Cartesian (ax, ay), mean-pooled, speed and displacement regressed out: ≈0 through block 12 (−0.008 … −0.003), 0.174 at 16, 0.456 at 22; leave-one-cell-out signed a 0.42 / 0.53 / 0.83 at 8 / 12 / 22 | results/p5_accel_decorrelated_cartesian_loco.json → models.vjepa2.points.{1,4,8,9,12,16,22}.meanpool.per_clip_folds.cartesian.mean.r2; points.{8,12,22}.meanpool.leave_one_cell_out.signed_a.r2 (0.420, 0.533, 0.830) |
| found notes, p2-shapes subtitle | frame wording: polar plus speed "fits best among hand-built frames"; no "native/own coordinate" | REPORT l.41; 4-D Fourier frame explains 0.33 / 0.29 of variance |

## Round 12 (matched-size probe-free comparison)

| Slide | Number | Source (file → key) |
|---|---|---|
| headline, found, p2-compare notes (p2-compare subtitle, found row "Curve steers heading") | block 22, all tokens, both edits at the twin's change size (median norm 19.07): spline vs chord forced choice 0.059 vs 0.094 (paired −0.035 [−0.052, −0.019]); recovery 0.186 vs 0.234 (−0.048 [−0.057, −0.040]); twin identification 0.331 vs 0.312 (+0.019 [+0.001, +0.037]); heading error 12.3° vs 25.7° (−13.4 [−15.6, −11.1]); n = 200 | results/session2_probefree_matched_size.json → matched.P22_natural.{spline,chord,spline_minus_chord}.{forced_choice_frac,recovery_full,twin_id_acc,heading_err_deg}; matched.P22_natural.edit_size.*.median (19.068) |
| (not on slides; for Q&A) | block 12 disk tokens (union), matched size: spline − chord forced choice +0.087 [0.043, 0.126], twin ID −0.024 [−0.045, −0.003], heading +59.4°; at the spline's own size forced choice +0.034 [−0.016, 0.081] | matched.D12_natural_union.spline_minus_chord.*; ladder.D12.spline_minus_chord_all_pairs.forced_choice_frac |
| limits notes | probe-free readouts reward edit size | ladder.P22 / ladder.D12 (rungs and chord_interp_at_spline_size) |

## Round 13 (QA fire #37)

| Slide | Number | Source (file → key) |
|---|---|---|
| p2-compare, found (singular "the spline", block 22) | Fourier-4 frame minus chord, block 22 heading error: +4.03° [2.28, 5.56] own size, +2.94° [1.01, 4.51] natural size (n = 200); the Fourier frame is worse than the chord | results/session2_fourier4_predictor.json → task1.paired.P22.{own,natural}.fourier_minus_chord.err |
| limits notes | block 12, disk tokens (union), matched dose: spline heading 100.2° vs chord 40.8° (spline − chord +59.4° [53.2, 65.8]) while spline forced choice leads, 0.163 vs 0.075 (+0.087 [0.043, 0.126]) | results/session2_probefree_matched_size.json → matched.D12_natural_union.{spline,chord}.heading_err_deg.mean (100.24, 40.85); matched.D12_natural_union.spline_minus_chord.{heading_err_deg,forced_choice_frac} |
| found, p2-compare notes | matched size (block 22): spline 12.3° vs chord 25.7° heading with recovery 0.186 vs 0.234; own size, separately: 11.3° vs 27.2° | matched.P22_natural.*; results/session2_fourier4_predictor.json → task1.table.P22.own.{spline,chord}.err_to_target.mean |
| limits notes | cache mismatch now quoted on the twin-change scale only (median ≈ 7%, max ≈ 1.3 twin changes) | as Round 11 row (lead's QA #397 figures) |

## Round 14 (backup slide p1-scaling; results/p5_scaling_layerwise.json, written 20:48)

| Slide | Number | Source (file → key) |
|---|---|---|
| p1-scaling title, notes, figure dots | direction CV onset (first point ≥ 90% of max CV R²): ViT-L block 2 (0.083 of depth, CI [2, 2]), ViT-H 3 (0.094, [3, 3]), ViT-g 4 (0.100, [4, 4]); all ≤ 0.10 = "first tenth" | models.{vitl,vith,vitg}_pretrained.variables.direction.summary_cv.{onset,onset_frac,onset_ci} |
| p1-scaling subtitle | ViT-L direction CV R² 0.875 at block 1 (test 0.888) on the supplied clips | models.vitl_pretrained.variables.direction.layers[1].{cv_mean,test_r2}; definitions.readability_not_zone |
| p1-scaling notes | untrained direction CV peaks: ViT-L 0.868, ViT-H 0.856, ViT-g 0.846 (quoted as 0.85–0.87) | models.{vitl,vith,vitg}_random.variables.direction.summary_cv.peak_score |
| p1-scaling title, line, notes, figure right | hard-render half-frame transfer: ViT-H (seed 0) −2.44 at block 8 → 0.40 at 9 (recovery point 9); second dip −0.33 at 12, −1.63 at 13; ViT-L −1.42 at 8 → 0.08 at 9 (recovery point 9) | zone_halfframe_hard.vith.{points,cross_half_r2,summary}; zone_halfframe_hard.vitl.{cross_half_r2,summary} (ViT-L source results/p1a_perpatch_direction_vjepa2_hard.json) |
| p1-scaling notes | bf16 parity on ViT-L only: median relative L2 error 0.52% (direction), 0.50% (speed), 0.53% (acceleration); min cosine 0.9957 | parity_vitl_bf16.{direction,speed,acceleration}.{rel_l2_err_median,cos_min} |
| p1-scaling notes | ~~ViT-g untrained acceleration/axay in flight at merge~~ SUPERSEDED: final merge (file 20:55) has vitg_random acceleration and axay | models.vitg_random.variables.{acceleration,axay} |
| p1-scaling notes | trained direction peaks flat after a third of depth: CV R² 0.973–0.991 (ViT-L), 0.985–0.989 (ViT-H), 0.971–0.992 (ViT-g); peak blocks 22 / 18 / 38 | models.*_pretrained.variables.direction.{layers[frac ≥ 1/3].cv_mean, summary_cv.peak} |
| p1-scaling notes | patch embedding (point 0) direction CV R²: trained ViT-H 0.709, ViT-g 0.718, ViT-L 0.100; untrained ViT-H 0.045 (ViT-L 0.058, ViT-g 0.058); unexplained | models.*.variables.direction.layers[0].cv_mean |
| p1-scaling figure | fig_p1_scaling.png (deck palette; x = block ÷ depth excluding the post-LN point; test R² curves, CV onset dots) | talk/make_talk_figs_v2.py scaling |

## Round 15 (user: six physics variables; Part 1 / Part 2 dividers; figure label collisions)

| Slide | Number / claim | Source (file → key) |
|---|---|---|
| p2-variables | "Which tokens" row removed (binding stays on p2-binding); six rows Direction, Speed, Acceleration, Position, Contact, Frame count; title "Six variables are readable…" | — |
| p2-variables Position (shape "plane") | start position occupies its own 2-D plane, orthogonal to direction within the null (block 12 dir|pos overlap 0.0017) | results/p5_motion_geometry.json → exp1_whitened_metric.points.vjepa2.12.pairs["dir\|pos"].overlap; REPORT subspaces paragraph |
| p2-variables Position (steered ●) | in-subspace position edit moves the start-position readout 5.2 / 7.5 natural spreads (speed set, blocks 12 / 22), 5.7 / 9.2 (acceleration set) | results/p5_motion_geometry.json → exp2_interference_leakage.sets.{speed,acceleration}.{12,22}.natural.pos.pos.mean |
| p2-variables Position (forecast) | changed from "yes" to "untested": no position edit was read through the predictor (only the encoder); REPORT reports forecast per-step position readouts as unreliable (−0.18 vs 0.91) | REPORT §4.x acceleration-through-predictor paragraph (l.1747); no results file for a position edit through the predictor |
| p1-scaling title, line, notes (QA #426) | ViT-H half-frame transfer after the first recovery: 0.398 at 9, −0.051 [−0.139, −0.014] at 10, 0.089 at 11, −0.332 at 12, −1.627 at 13, 0.406 at 14 and positive after; durable recovery 14 = 0.4375 of depth vs ViT-L 9 = 0.375; first recovery 9 in both (0.375 / 0.281) | results/p5_scaling_layerwise.json → zone_halfframe_hard.vith.{points,cross_half_r2,cross_half_r2_ci95}; zone_halfframe_hard.{vitl,vith}.summary.recovery_point |
| p1-scaling notes (QA #428) | patch-embedding values labelled as CV R² (0.709 / 0.718 / 0.100 / 0.045); test R² 0.739 / 0.740 / 0.035 / 0.010 (ViT-H, ViT-g, ViT-L trained; ViT-H untrained) | models.{vith,vitg,vitl}_pretrained, vith_random .variables.direction.layers[0].{cv_mean,test_r2} |

## Round 15 add-on (backup slide p2-heads; results/p5_motion_heads.json, landed 21:01 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p2-heads title, notes | 125 of 128 heads (blocks 6–13) have previous-slot disk / background density > 5 (127 > 2); block medians 37.5–75.0; top 6 hold 32% of the excess | part_A.vjepa2.{n_heads_ratio_gt_5, n_heads_ratio_gt_2, block_median_ratio, top6_of_128_share_of_excess} |
| p2-heads notes | phantom (mask rolled 8, 8 patches) block medians 1.07–8.95 | part_A.vjepa2.block_median_phantom_ratio |
| p2-heads notes, figure rings | 19 heads under the weak label, 9 under the strict one (prev beats same, next and far with prev/far CI > 1; phantom below real: b6h7, b7h15, b8h14, b9h4, b10h4, b11h6, b12h3, b13h3, b13h8), three with prev/same above 10 (b7h15 13.32, b11h6 13.27, b13h8 13.31; blocks 7, 11, 13), none of those three ablated; phantom/real 0.58 / 0.29 / 0.12. Of the 9, four sit in ablated blocks and were zeroed singly (b8h14 0.011, b9h4 0.008, b10h4 0.017, b12h3 0.016 block-12 drop); only b10h4 is in the joint top 8 | part_A.vjepa2.table[{7,15},{11,6},{13,8}].{prev_over_same_density,next_over_bg_density,far_over_bg_density,phantom_prev_over_bg_density}; top_tracking_heads_joined[].direction_p12_drop = null |
| p2-heads notes | weak "tracking" label (prev/same CI > 1): 19 of 128 trained, 69 untrained at ratios 0.76–1.51, so not evidence | part_A.vjepa2.n_heads_labelled_tracking; part_A.random.{n_heads_labelled_tracking, table[].prev_over_bg_density} |
| p2-heads line, notes, figure right | joint top-8 ablation (b9h9, b12h6, b10h4, b8h2, b10h12, b12h13, b10h6, b8h15): block-12 direction R² drop 0.040 [0.035, 0.045] vs random count-matched 0.207 / 0.123 / 0.174 / 0.220; block 22: 0.0051 [0.0037, 0.0066] vs 0.0019–0.0066 | part_B.vjepa2.direction.{p12,p22}.drops.{global_top,global_rand0..3}; part_B.vjepa2.plan.top_global |
| p2-heads (Q&A) | largest single head b12h4 0.312 at point 12; ≤ 0.0054 at point 22; speed top-8 0.059 vs random 0.009–0.582 | part_B.vjepa2.summary.direction_p12.12.max_head_drop; direction_p22.*.max_head_drop; speed_p12.global |
| p2-heads notes | unablated baseline 0.9882 vs stored fp32 0.9885; untrained ablation invalid in bf16 (unablated R² −15.74); untrained attention max ratio 1.51 | part_B.vjepa2.direction.p12.baseline; part_B.random.direction.p12.baseline.test_r2_box_unablated; part_A.random.block_max_ratio |
| p2-variables Position (forecast) | see Round 15 table above | — |

## Round 16 (assessor pass; QA fire #40 rows #450–#456)

| Slide | Number | Source (file → key) |
|---|---|---|
| headline (caption "paper's spline, one arc"; line "Our smoother 44°, chord 27°"), p2-splines notes | block 22 through the predictor, own norm: paper's interpolating spline 11.3° [10.4, 12.2]; our smoothing spline 44.2° [39.8, 48.8]; chord 27.2° [24.4, 30.5]; n = 200 | results/session2_predictor_native_readout.json → per_layer.22.{spline,spline_smooth,chord}.dir_err_to_target |
| p2-splines notes | smoother diagnosis: aim point reads 4.3° off vs 1.5° for the chord (block 22); edit ‖Δ‖ ÷ twin change 0.62 vs 0.91 for the interpolating spline | REPORT §4.3 "Why the spline lost the held-out endpoint" (l.998–1003); REPORT predictor table last column (l.1497–1499) |
| limits (visible "twin cache 9.1° vs 9.26°") | twin ceiling 9.107° cached (session-2 forecast) vs 9.258° recomputed | results/session2_fourier4_predictor.json → task1.positive_control_twin_forecast_session2_cache_vs_rerun_reference.err_to_target.mean (9.107); task1.twin_ceiling.err_to_target.mean (9.258); REPORT l.2111 |
| p2-compare (line "Long way round: spline radius 0.79, raw chord 0.055"), notes | antipode-in-arc, 16 arcs, block 22: spline route (+90°) minimum probe radius 0.789 [0.767, 0.811], intermediate mass 0.958 [0.948, 0.968]; raw chord minimum radius 0.055 [0.052, 0.058] (smoothed-knot chord 0.068); no intermediate mass is stored for chord routes | results/p2_two_route_heldout_L22.json → over_arcs.antipode_in_arc.via_plus90.{probe_radius_min,probe_intermediate_mass}; over_arcs.antipode_in_arc.chord_raw.probe_radius_min (chord.probe_radius_min 0.068) |
| p2-compare subtitle "At held-out endpoints the chord lands as close as any curve", notes; found, close, vs-paper notes (#450) | paired endpoint error, curve − raw chord (positive = chord closer): paper's interpolating spline +0.81 [0.74, 0.89] (22), +27.87 [27.39, 28.35] (12, label-free angle); our smoother +2.26 [2.09, 2.42] (22), +1.48 [1.27, 1.69] (12); exploratory knot-CV smoother −0.45 [−0.50, −0.40] (22), −0.38 [−0.50, −0.27] (12) | results/p2_bakeoff_unified_16arc.json → paired_vs_chord["{interp_causalab_lam0,fitpack_smooth,causalab_lam_cv} - chord_raw"].err_end.{22,12} |
| p2-compare notes (#452) | COAST as written (full 1,024-d, α 0.1, uncentred), block 12: 88.1° at the chord's norm (unsteered 88.9°, chord 4.7°), 81.4° at own norm (β 0.3), 55.5° (β 1); our PCA-64 variant 114.4° (β 0.3) | results/p2_coast_faithful_L12.json → arms["full\|contr\|jaeger\|a0.1\|b0.3\|unc"].{probe_chordnorm,probe_own}; arms["full\|contr\|jaeger\|a0.1\|b1\|unc"].probe_own; arms["pca64\|contr\|jaeger\|a0.1\|b0.3\|unc"].probe_own; unsteered_err.probe. (The L22 file gives 89.6 / 93.0 / 103.3 for the full-space arm; the notes quote block 12) |
| p2-repair notes (#451) | survival at block 24: random edit 0.020 vs spline 0.091 / raw chord 0.107 (block-12 edits, 16 carriers): correction is generic | results/p5_repair_attribution_L12.json → attribution_summary.{random12,spline12,rawchord12}.survival_final_block24 |
| p2-coordinates notes (#455) | post-hoc constant-velocity subset at the chord's norm: polar with second harmonic 6.10 / 6.14 vs x–y 8.12 / 8.74 (blocks 12 / 22); x–y beats 2-D polar by 0.41 / 1.58 | results/p5_coordinate_competition.json → steering_posthoc_velocity_carriers.{12,22}.arms.{polar4,cartesian}@chord_norm.err_end_deg.mean; cartesian_minus_polar2.chord_norm.mean |
| p2-shapes notes (#456) | encoder, block 22, joint: smoothing sheet 4.33 vs interpolating 1-D curves in turn 6.98; like for like interpolating sheet 4.67 vs 6.98 (smoothing sheet vs smoothing 1-D 4.33 vs 8.20) | results/p5_velocity_sheet_v2.json → results.22.block2.joint.own.summary.{sheet,sheet_tps_interp,seq_global_interp,seq_global}.err_dir.mean |
| p1-zone title | hard-render half-frame transfer, three-seed means: 0.708 (block 1), −0.136 (4; seeds 1, 2 still positive 0.162, 0.154), −0.362 (6), −1.082 (7), −1.432 (8; all seeds negative at 7 and 8), 0.181 (9), 0.303 (10), 0.580 (12) | results/p1a_perpatch_hard_seeds.json → by_point[].cross_half_r2.{mean,per_seed} |
| found notes (item 5) | acceleration beyond mean speed on decorrelated clips ≈0 (blocks 1–4), 0.37 (8), 0.79 (22); supplied clips start at rest | as Round 10 rows (results/p5_accel_decorrelated.json) |

## QA #41 (fire at 21:47 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p2-compare line, notes | antipode, either 180° route: smoother MLP ordering 0.966 [0.957, 0.974] (via +90) / 0.975 (via −90); raw chord MLP ordering 0.666 / 0.688 per half; chord intermediate mass 0.360 / 0.370 (metric has a floor from endpoint scatter); radius floor 0.055 forced by linearity | results/p2_two_route_heldout_L22.json → over_arcs.antipode_in_arc.{via_plus90,via_minus90}.mlp_ordering; chord_raw.{mlp_ordering_plus_half,mlp_ordering_minus_half,probe_intermediate_mass_*_half,probe_radius_min} |
| p2-compare footer | 16 seeds over 15 distinct arcs (seeds 4 and 8 draw the same arc, block 46); figure legend still says 16 arcs (QA #461) | results/p2_bakeoff_unified_16arc.json arcs |
| p2-splines notes | smoother 44.19 vs chord 27.25 is not size: edit 0.62 vs 0.65 of the twin change; 39.13 at the chord's norm; encoder miss on this arc at block 22, 10.72 vs 3.59 at norms 12.40 vs 12.68 (QA #462) | results/session2_predictor_native_readout.json per_layer.22.{spline_smooth,chord} |
| found notes | acceleration beyond mean speed per tubelet 0.53 at block 1 (pooled ≈0) (QA #464) | results/p5_accel_decorrelated.json timepool block 1 |

## QA #42 (fire at 22:05 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p1-zone title, alt, notes | half-frame transfer per seed: block 1 0.69 / 0.73 / 0.71; blocks 7 and 8 negative on all three seeds; block 9 0.077 / 0.385 / 0.082 (mean 0.18, chance); mean 0.58 at 12, 0.76 at 22; block 4 +0.16 / +0.15 on seeds 1, 2 (QA #466) | results/p1a_perpatch_hard_seeds.json → by_point[].cross_half_r2.per_seed |
| p1-zone footer, notes | partial reproduction: supplied clips transfer 0.82 at point 1 rising to 0.95 with no dip; untrained 0.61 → 0.74 (QA #467); MAE 56° vs 11° offset caveat (QA #468); per-patch R² 0.95 vs the paper's Table 4 0.72 (QA #469) | results/p1a_perpatch_direction_vjepa2.json, results/p1a_perpatch_direction_random.json; REPORT §3.1 |
| headline caption, line | 11.3° = interpolating spline (Goodfire A.3, language-task recipe); 44.2° = our smoother on their B.1 world-model recipe (QA #472) | results/session2_predictor_native_readout.json per_layer.22.{spline,spline_smooth} |
| papers column, notes | Musa et al. 2026 (Joseph's group): spline steering inside V-JEPA 2, in sample (QA #471); §7 l.390–395 poses the use question; §6.3 tests how the code forms (QA #470) | refs/physics_paper.txt; literature review notes (not included) |

## QA #43 (fire at 22:19 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p1-reproduce title, notes; p1-steer notes | 18 Adam probes judged by our ridge α = 100 probe; judge recipe alone moves the ridge count 5 → 3 (α = 1e-3) / 4 (Adam judge); Adam basis × Adam judge not yet scored (rerun in progress, results/p1c_direction_L9_adam_judge.json) (QA #475) | results/p1c_direction_L9_adam_basis.json eval_probe.alpha; results/p1c_direction_evalprobe_recipe.json recipe_results.*.n_to_10deg |
| p1-steer notes, vs-paper cell | judge fit on the unsteered test clips and read on their steered versions (C.12 l.1245); split-half 14.7° (QA #476) | src/wm/steer.py:run_steering eval_probe_cv(Xte, Y[te]) |
| p1-zone title, alt, notes | block 9 per seed 0.077 / 0.385 / 0.082 (no permutation baseline, not "chance"); block 12 0.42 / 0.69 / 0.63 below block 1 on every seed; 16 0.36 / 0.58 / 0.60; block-1 level back at 22 (0.76); loss begins block 4 (seed 0) / 5 (seeds 1–2) (QA #477, #478); Table 4 comparison removed (QA #479) | results/p1a_perpatch_hard_seeds.json by_point[].cross_half_r2.per_seed |
| headline line, papers title | "Our smoother" (B.1-style, departs from B.1); "Prior work … none asked" (three author groups) (QA #480) | — |

## #475 rerun (ce3123d, 22:25 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p1-reproduce title, notes; p1-steer notes | n_to_10deg: ridge basis 5 / 3 / 4 / 6 under ridge α100 / ridge α1e-3 / Adam C.11 judge / split-half; Adam basis 18 / 11 / 11 first crossing, 19 sustained / 22; at N=5 under the Adam judge ridge 8.0° beats all 20 same-rank random draws (median 86.6°, p 1/21), Adam 72.2° vs 79.4° (p 0.38) | results/p1c_direction_L9_adam_judge.json → per basis × judge n_to_10deg, sustained, mae_at_n, random_null |

## QA #44 (fire at 22:29 ET)

| Slide | Number | Source (file → key) |
|---|---|---|
| p2-compare line, notes | chord_raw MLP offset by waypoint (antipode route): 10.6°, 16.6°, 64.9°, 154.6° at waypoints 24–27; spline 2.5° → 177.7° monotone (QA #483) | results/p2_two_route_heldout_L22.json per_waypoint.antipode_in_arc.chord_raw.mlp_offset_circmean; via_plus90 |
| p2-compare notes; headline notes | all deck edits additive (shift + residual kept); under the paper's A.6 replace rule over 16 seeds the chord leads the paper's spline 1.63 vs 4.83 (3.2°) at block 22 and 1.50 vs 6.80 (5.3°) at block 12 (QA #482) | results/p2_endpoint_diagnosis.json decomposition.*.paper_protocol |
| p2-compare notes | along-path forecast: spline − chord −13.40° [−15.66, −10.99]; min forecast radius 0.528 spline vs 0.258 chord; size matched at the endpoint only (QA #484) | results/session2_predictor_along_path.json |

## Deck v77 (28 Sep, late): tables simplified on the user's request

| Slide | Change | Source |
|---|---|---|
| p2-variables | cells are now one key only (● yes · ◐ in part · ○ no · — not tested); no coloured words. Speed and acceleration "moves the forecast" ◐ (speed code moves, disk does not; acceleration 1.82 vs 3.46 unedited, twin 0.83); Position —; Contact steerable ○ (heading control 0.77 vs 0.48), forecast ◐ (seen bounce 0.84, none anticipated); Frame count ○ | same rows as above (#197, #226, #238, #258, #266) |
| cites | rebuilt as a three-column grid (paper, title, used here for); no numbers | — |
| p2-compare | bake-off figure regenerated with legend "across 16 seeds, 15 arcs" (seeds 4 and 8 coincide); new blob 2254de541a0522e8a4667a6460791d07; alt text and limits notes say 16 seeds over 15 arcs | results/p2_bakeoff_unified_16arc.json (unchanged) |
| p2-variables (v79, QA #492/#493) | column header "forecast uses it"; frame count ○ means not used as time (the push moves the disk 15.5 px sideways and turns it 47.9°, notes); contact shape cell blank | results/p5_time_patch_predictor.json → arms.22.tp+2 |
| p1-steer (v79, QA #494) | "same-rank random directions do not" (6 of 200 draws reach the ridge value at five probes, p 0.035) | results/p1c_direction_L9_nulls200.json → bases.ridge.table[n=5].n_draws_le_learned |
| p2-variables (v80) | header cell "variable" and every grid cell non-empty (the live viewer collapses empty cells); contact shape "binary"; three lesson lines under the table: position plane edited in the encoder, untested through the forecast (REPORT §4, position sheet; p5_motion_geometry dir|pos overlap 0.0017); contact seen 0.84 / unseen 0.07, edit writes a heading 0.77 vs 0.48; frame count slot index (untrained too), push does not advance the forecast | results/p5_contact_in_context.json; p5_contact_steer_heading_control.json; p5_time_shuffle.json; p5_time_patch_predictor.json |
| p1-zone (v80, QA #501) | title: "is near zero at 9 and back only at 22" (per-seed block 9: 0.08 / 0.39 / 0.08; block 22: 0.76) | results/p1a_perpatch_hard_seeds.json → by_point |
| p2-variables (v81) | one six-row table: variable, shape, what the forecast does with it. Direction 11.3° vs 92° (session2_predictor_native_readout); speed code moves, displacement not (session3_speed_predictor); acceleration 1.8 vs 3.5 (session3_acceleration_predictor 1.818 / 3.462); position edited in the encoder, forecast untested; contact 0.84 / 0.07, edit writes a heading (0.77 vs 0.48); frame count slot index, untrained encoder too, push does not advance (p5_time_shuffle, p5_time_patch_predictor) | as listed |
| p2-vs-probe (new, v82) | plain-English comparison with the Part 1 multi-probe subspace: endpoint tie (probe 4.1° / 4.2° at blocks 12 / 22 vs paper's spline 5.0°, smoother 6.5° at 22); path: probe and chord edits fall to 0.59–0.63 of a real clip's heading, curves keep ≥0.80; cost: probe arm ties only at 1.5× the edit size, at equal size 22.8° (R 0.00); both repaired at block 12 unless on the disk's tokens | results/p2_bakeoff_unified_16arc.json table.{probe_qr,chord_raw,interp_causalab_lam0,fitpack_smooth}.{err_end,radius_min}; p2_endpoint_diagnosis.json (probe steer at the chord's norm); p5_repair_attribution_L12.json |
| p2-compare (v82) | antipode line in plain English: "To the opposite heading, the curve's readout turns steadily; the straight edit's stays put, then jumps." (waypoint offsets 11°, 17°, 65°, 155° stay in the notes) | results/p2_two_route_heldout.json |
| headline (v84) | title is the causal claim in plain words: edit the heading where the predictor reads it and the forecast turns; edit it mid-depth and the encoder undoes it first. Numbers unchanged: 11.3° after the block-22 edit, 92.1° no edit, 9.1° real clip; bold line block-12 frame-wide 80° (79.7) with 9% of the edit left, disk tokens 45° (44.7) | session2_predictor_native_readout.json; session2_direction_natural_norm.json; p5_repair_attribution_L12.json |
| p2-vs-probe (v84) | qualitative verdicts: same edit on a line; tie at the target on the ring (probe 4.1 / 4.2 vs paper's spline 5.0); curve wins on the way (0.59 vs 0.80 radius; forecast follows the curve, along-path lead 13°); moot at mid depth; footer gives the three rules for steering a world model | as in the v82 row |
| p2-variables (v85) | no numbers on the slide: columns are variable, shape of its code and why (ring: class means close into a loop ordered by heading, circular correlation −0.98; line: means on one straight axis; acceleration equals mean speed on the supplied clips; plane: x and y axes orthogonal to the motion code, overlap 0.0017; event: a turn in the heading code after the bounce, 0.80 after vs 0.22 before; slot index: slot coefficient 0.96, untrained too) and what the forecast does (phrases only); numbers in the notes | p2_geometry_direction_L12; p2_bakeoff_{speed_L19,acceleration_L21}; p5_motion_geometry; p5_contact_in_context; p5_time_shuffle; session2_predictor_native_readout; session3_*_predictor |
| headline (v87) | reframed against both papers: "The bridge from representation to behaviour exists only at the encoder's exit: an edit in the physics emergence zone is undone before the predictor sees it"; shows fig_read_vs_use (blob cbdf16e1763a7a208c27d9b629c2a828; top per-patch R² by block, bottom forecast error after an edit at each block: 92.1 unedited, 11.3 at block 22); no numbers in the text, all in the notes (11.3 / 92.1 / 9.1; block 12 9% left, 80°, disk tokens 45°; smoother 44, chord 27; probe-free 0.19–0.26) | session2_predictor_native_readout.json; session2_direction_natural_norm.json; p5_repair_attribution_L12.json; session2_disk_token_sweep.json; session2_probefree_matched_size.json |
| v89 | limits | "a fifth of the way" (probe-free recovery 0.15–0.26), "fifteen held-out arcs", "one arc through the predictor", "two untrained seeds", "block 12" disk-token edit reaches the forecast partway | session2_probefree_matched_size.json; p2_bakeoff_unified_16arc.json (16 seeds, 15 distinct arcs); session2 200×4 on one 45° arc; p1 untrained seeds 0/1; session2_disk_token_sweep*.json (44.7° vs 92.1°) | numbers in the notes only; the slide is qualitative |
| v91 | rigor (new backup) | 1,200 / 300 clips, five folds; 16 seeds over 15 arcs × 8 targets × 48 clips, 1,000-draw paired clip bootstrap, resolves 0.1–0.2°; 200 carriers × 4 targets, 1,000-draw carrier bootstrap, about 2.5°; position 48 × 2; INLP K 33–42; multi-probe null 200 draws; token patching 64 + 64 pairs | REPORT §7.1; p2_bakeoff_unified_16arc.json; session2_predictor_native_readout.json; session2_position_predictor.json; p5_token_patching.json | seven edit arms labelled linear / geometric |
| v91 | p2-variables | Position cell "moves after a late edit; a mid-depth edit is undone"; notes: block 22 closes 0.35 of the gap (22 px vs 34 unedited, twin 6), block 12 0.01, encoder-exit R 0.009 | session2_position_predictor.json per_layer.{22,12}.{sheet,chord}.forecast.{frac_distance_closed,dist_to_target_px}, reference.*, encoder_full_clip.24 | QA #512: contact source is p5_contact_dynamics.json |
| v91 | headline | title "opens only in the encoder's last blocks ... mostly undone" (QA #506); notes: block 12 forecast 80° off (R 0.3); probe-free 0.19–0.26 at natural size (#513); position 0.35 / 0.01 | session2_direction_natural_norm.json table.12 (chord 79.7, R 0.29–0.31 spline); session2_fourier4_predictor.json task1.table.P22.natural; session2_position_predictor.json | |
| v91 | p2-vs-probe | "The curve adds nothing" on lines (probe edit cosine 0.38–0.52 with the spline; 0.526 vs 0.092 speed, 0.637 vs 0.282 acceleration at matched norm, QA #505); "A tie in the encoder" (through the predictor at block 22: curve 11.3°, probe 17.1°, chord 27.2°, QA #507); "almost no clip sits" (5.5% below radius 0.5, #514); 1.5× push at block 22 only (#508) | p2_cosine_tangent_{speed_L19,acceleration_L21}_contiguous.json; p2_bakeoff_{speed_L19,acceleration_L21}_contiguous.json arms.*.err_probe_matched; session2_predictor_native_readout.json per_layer.22; p2_ring_occupancy_L22.json | |
| v91 | limits | notes: probe-free 0.19–0.26 natural / 0.15–0.17 own (#513); source p2_endpoint_diagnosis.json (#511); position landed 0.35 / 0.01 | session2_fourier4_predictor.json; p2_endpoint_diagnosis.json; session2_position_predictor.json | footer no longer says the run is in progress |
| v92 | headline | title "Steering in the Physics Emergence Zone moves the probe, not the forecast; steering the last blocks moves both"; subtitle in the papers' terms (held-out probe on steered activations, C.12; behaviour manifold); notes state that every edit is in the encoder and the predictor / V-JEPA 2-AC are never edited | same sources as v91 | wording only |
| v93 | headline | subtitle carries the headline numbers: block-22 spline 11.3° (shown as 11°), chord 27.2° (27°), real clip 9.1° (9°), block-12 79.7°/77.9° (80°) | session2_predictor_native_readout.json per_layer.22.{spline,chord}.dir_err_to_target, twin 9.107; per_layer.12.{chord 84.8 native-readout; natural-norm chord 79.7 / spline 77.9} | title "Manifold steering at block 22 changes what V-JEPA 2 forecasts; in the Physics Emergence Zone the same edit is undone first" |
| v93 | design | legend as plain cells with circled digits and blocks; zone label "Physics Emergence Zone (Joseph et al.): the one-third-depth transition; our tests at blocks 8–9"; predictor label "a separate transformer after the encoder" | physics paper l.232–234 ("consistent one-third depth transition"), l.1011 ("sharp transition between Layers 7 and 8") | no numbers changed |
| v93 | rigor | restyled (padded rows, shorter cells); same numbers as v91 | as v91 | |
