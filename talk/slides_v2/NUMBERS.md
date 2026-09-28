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
| p2-clock (figure + rail) | V-JEPA 2 0.99 / 1.00 / 1.00 per step, CI [0.983, 0.994] / [0.993, 0.998] / [0.994, 0.998] | results/p5_time_manifold_controls.json → ["speed/vjepa2/timepool_same_subset"].clock.22.spline.{slow,mid,fast}.slope.{mean,ci95} (0.9885, 0.9959, 0.9960); n_clips 100 / 135 / 63 |
| p2-clock (figure + rail) | untrained copy 0.71 / 0.74 / 0.77, CI [0.63, 0.80] / [0.69, 0.80] / [0.71, 0.83] | same file → ["speed/random/timepool"].clock.22.spline.{slow,mid,fast}.slope.{mean,ci95} (0.7128, 0.7432, 0.7659) |
| p2-clock (figure + rail) | distance counter 0.21 / 1.01 / 1.59 (hollow ochre) | same file → ["speed/vjepa2/timepool_same_subset"].clock.22.spline.{slow,mid,fast}.odometer_prediction (0.2144, 1.0078, 1.5945); identical under speed/random/timepool |
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
| p2-binding notes | per-patch R² 0.975 at block 12 | results/p1a_perpatch_direction_vjepa2.json → curves.perpos_mean_r2 at point 12 (0.975) |
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
| p2-coordinates (big number), found | 0.76 vs 0.46 at block 12; winner at 6 of 7 blocks; untrained best x–y 0.34–0.53 | results/p5_motion_geometry.json → exp4_coordinate_search_and_exp5_shortcuts.points.12.per_coordinate.{fourier2_polar,cartesian}.encoding_r2_transfer_to_dirset (0.763, 0.456); winner_by_point (fourier2_polar at 1, 8, 12, 16, 19, 22; fourier_logpolar at 4); random_init.*.cartesian.encoding_r2_transfer_to_dirset (0.340–0.534, best at every point) |
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
| p2-shapes notes | sheet − ring-only through the predictor: +1.54 [−1.93, 5.13] own norm; +6.72 [2.82, 10.38] at the sheet's norm (n = 48) | results/p5_velocity_sheet_predictor.json → paired.{own,sheet_norm}.sheet_minus_ring1d_dir_err |

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
