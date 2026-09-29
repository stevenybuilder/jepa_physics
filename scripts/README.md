# scripts/

One entry point per experiment. Every script runs from the repo root (`python scripts/<name>.py`, most take `--help`)
and imports the library from `src/wm` without installation. Scripts marked **GPU** re-encode clips with the model
(V-JEPA 2 ViT-L; a CUDA box in practice); everything else runs on CPU from `artifacts/` activations or from
`results/`. `artifacts/` (activations, stimuli, forward caches) is not in the repository: regenerate it with
`extract.py` and the GPU scripts. REPORT § is where the output is used.

Scripts that load the Goodfire authors' spline code expect `git clone https://github.com/goodfire-ai/causalab refs/causalab`
(commit `1b6f43a`): `run_endpoint_diagnosis*.py`, `run_bakeoff_unified_16arc.py` and the scripts that import them.

## Data, extraction, splits

| script | produces | REPORT § |
|---|---|---|
| `run_qa.py` | `results/qa_{dataset}.json` (frame hashes, label checks) | 2 |
| `make_splits.py` | `splits/split_v1.json` (reads `results/qa_*.json`; `$WM_SPLIT_PATH`, `$WM_TEST_SIZE` for the 70/30 split) | 2 |
| `extract.py` **GPU** | `artifacts/activations/{dataset}/{model}/` (meanpool, timepool, index) | 2, 3.1 |
| `check_parity.py` | CPU-vs-GPU extraction parity JSON (`--json`), recorded in `artifacts/gpu_session1.json` | 2 |
| `check_probe_recipe.py` | `results/p1a_probe_recipe_check.json` (paper Adam probe vs ridge) | 2 |
| `p1a_perpatch.py` **GPU** | `results/p1a_perpatch_direction_{model}{suffix}.json` | 3.1 |

## Part 1: probing, INLP, subspace steering

| script | produces | REPORT § |
|---|---|---|
| `run_step1.py` | `results/p1a_{dataset}_{variable}_{pool}[_shuffled].json` | 3.1 |
| `run_step1_paperscale.py` | `results/p1a_paperscale_{dataset}_{variable}.json`, `figures/fig1f_paperscale.png` | 3.1 |
| `run_step1_support.py` | `results/p1a_support_{onset,transfer,spatial}_*.json` | 5 |
| `run_paperscale_velocity.py` | `results/p1a_paperscale_velocity_only.json` | 3.1 |
| `run_appb_sweep.py` | `results/p1a_appB_sweep.json` | 3.1 |
| `run_attentive_probe.py` **GPU tokens** | `results/p1a_attentive_{var}.json` | 3.1 |
| `run_sigmoid_onset.py` | `results/p1a_sigmoid_onset.json` | 3.1 |
| `run_random_feature_floor.py` | `results/p1a_baseline_comparison.json` | 3.1 |
| `run_pixel_baseline.py` | `results/p1a_{dataset}_{variable}_{pixels,trajectory}.json` | 3.1 |
| `run_audit_robustness.py` | `results/p1a_grouped_cv.json`, `results/p1b_raw_coordinates.json`, `results/p1b_sawtooth_metrics*.json` | 3.1, 3.2, 7.1 |
| `run_step2.py` | `results/p1b_{dataset}_{variable}_{pool}_L{L}[_adam].json`, `artifacts/inlp/` bases | 3.2 |
| `run_step2_dims.py` | `results/p1b_dims_four_ways.json` | 3.2 |
| `run_step2_angles.py` | `results/step2_subspace_angles.json` | 5 |
| `run_stop_rules.py` | `results/p1b_stop_rules.json`, `results/p1b_one_column_removal.json` | 3.2 |
| `run_sameclip.py`, `run_sameclip_adam.py` | `results/p1b_sameclip[_adam]_direction_vs_speed.json` | 3.2 |
| `run_step3.py` | `results/p1c_{dataset}_L{layer}[_rankmatched].json` | 3.3 |
| `run_step3_adam_basis.py` | `results/p1c_direction_L9_adam_basis.json` | 3.3 |
| `run_step3_adam_judge.py` | `results/p1c_direction_L9_adam_judge.json` | 3.3 |
| `run_step3_nulls200.py` | `results/p1c_direction_L9_nulls200.json` | 3.3 |
| `run_step3_rank2_covweighted.py` | `results/p1c_direction_L9_rank2_covweighted.json` | 3.3 |
| `run_scaling_layerwise.py` **GPU** | `results/p5_scaling_layerwise.json`, `figures/fig_scaling_layerwise.png` | 3.5 |
| `run_objective_axis.py` | `results/objective_axis.json`, `figures/fig5_objective_axis.png` (V-JEPA 2 vs VideoMAE vs random init) | 3.1, 3.3 |
| `fig_perpatch_videomae.py` | `figures/fig1k_perpatch_videomae.png` | 3.1 |

## Part 2: geometry, splines, bake-off

| script | produces | REPORT § |
|---|---|---|
| `run_geometry_checks.py` | `results/p2_geometry_{tag}.json`, `results/p2_planted_ring_{tag}.json` | 4.1 |
| `run_ellipse.py` | `results/p2_ellipse_direction.json` | 4.1 |
| `run_fft_harmonics.py` | `results/p2_fft_harmonics.json` | 4.1 |
| `run_velocity_plane.py` | `results/p2_velocity_plane.json` | 4.1 |
| `run_rotating_speed_axis.py` | `results/p2_rotating_speed_axis_{tag}.json` | 4.1 |
| `run_two_route.py`, `run_two_route_heldout.py` | `results/p2_two_route_direction_{tag}.json`, `results/p2_two_route_heldout_L{L}.json` | 4.1 |
| `run_ring_occupancy.py` | `results/p2_ring_occupancy_L{12,22}[_labels22].json` (via `--out`) | 4.3 |
| `run_part2.py` | `results/p2_steer_{dataset}_{variable}_L{layer}_{holdout}.json`, `figures/fig4_*` | 4.3 |
| `replot_part2_context.py` | refreshed `figures/fig4_*_{tag}.png` for a stored context run | 4.3 |
| `run_bakeoff.py` | `results/p2_bakeoff_{tag}.json` | 4.3 |
| `run_bakeoff_unified_16arc.py`, `aggregate_bakeoff_unified_16arc.py` | raw arcs in `results/endpoint_diagnosis_raw/`, then `results/p2_bakeoff_unified_16arc.json` (headline bake-off) | 1, 4.3 |
| `run_endpoint_diagnosis.py`, `summarize_endpoint_diagnosis.py` | `results/p2_endpoint_diagnosis.json`, `figures/fig_p2_endpoint_diagnosis.png` | 4.3 |
| `run_endpoint_diagnosis_path.py`, `aggregate_endpoint_diagnosis_path.py` | `results/p2_endpoint_diagnosis_path.json`, `figures/fig_p2_path_by_spline.png` | 4.3 |
| `summarize_shift_dependence.py` | `results/p2_shift_dependence.json` | 4.3 |
| `summarize_parity_scalars.py` | `results/p2_parity_scalars.json`, `results/p2_shift_dependence_scalars.json` | 4.3 |
| `summarize_p2_audit.py` | `results/p2_angle_source_audit.json`, `results/p2_interp_labels_summary.json`, `results/p2_extrapolation_linear_ext.json` | 4.1, 4.3 |
| `run_angle_goodfire.py` | `results/p2_angle_goodfire_method.json`, `results/p2_angle_goodfire_all64.json` | 4.1, 4.3 |
| `run_local_density.py` | `results/p2_local_density.json` | 4.3 |
| `run_offtarget.py` | `results/p2_offtarget_direction_L{layer}.json` | 4.3 |
| `run_donor_ceiling.py` | `results/p2_donor_ceiling_direction_{tag}_contiguous.json` | 4.3 |
| `run_cosine_tangent.py` | `results/p2_cosine_tangent_{variable}_L{point}_contiguous.json` | 4.3 |
| `run_isometry_linear.py` | `results/p2_isometry_{linear,goodfire_*}.json` | 4.4 |
| `run_conceptor.py`, `aggregate_conceptor_16arc.py` | `results/p2_conceptor_direction_L{L}.json`, `results/p2_conceptor_direction_16arc_L{L}.json` | 4.3, 4.4 |
| `run_coast_faithful.py`, `aggregate_coast_faithful.py` | `results/p2_coast_faithful_L{L}.json` | 4.4 |
| `run_geodesic.py`, `run_geodesic_device.py` (device-agnostic copy) | `results/p2_geodesic_direction_L{layer}{tag}.json` | 4.4 |
| `run_coordinate_competition.py` | `results/p5_coordinate_competition.json` | 4.4 |
| `plot_ring_fitted.py` | `figures/fig_ring_fitted_L{L}.png` from `results/p2_spline_validation_fig.npz` | 4.1 |

## Predictor tests (GPU session 2 and follow-ups)

| script | produces | REPORT § |
|---|---|---|
| `session2_box.sh` | driver: plan locally, push to a box (`HOST`, `PORT` env), run, pull, score | 4.5 |
| `run_session2.py` **GPU** (`forward`, `timerev`, `extract-stimuli`) | `results/session2_{plan,propagation,predictor,timerev}.json` | 4.5 |
| `session2_native_readout.py` | `results/session2_predictor_native_readout.json` | 4.5 |
| `session2_norm_matched.py` **GPU** | `results/session2_predictor_norm_matched.json` | 4.5 |
| `session2_along_path.py` **GPU** | `results/session2_predictor_along_path.json`, `results/session2_reverse_path.json` | 4.5 |
| `session2_extras.py` | `results/session2_future_position.json`, `results/session2_timerev_speed.json` | 4.5 |
| `session2_encoder_output.py` **GPU** | `results/session2_encoder_output.json`, `results/session2_interp_labels_L12.json` | 4.5 |
| `session2_direction_natural_norm.py` **GPU** | `results/session2_direction_natural_norm.json` | 4.5 |
| `session2_disk_token_sweep.py`, `session2_disk_token_sweep_twinfree.py` **GPU** | `results/session2_disk_token_sweep.json` | 4.5 |
| `session2_fourier4_predictor.py` **GPU** | `results/session2_fourier4_predictor.json` | 4.4, 4.5 |
| `session2_probefree_matched_size.py` | spline-minus-chord at matched size from cached forecasts | 4.5 |
| `session2_twin_difference.py` **GPU** | twin-difference arm, read into the predictor readouts | 4.5 |
| `session2_pullback_goodfire.py`, `session2_pullback_goodfire_ctx.py`, `session2_pullback_ctxring.py` | `results/session2_pullback_goodfire{,_ctx,_ctxring}.json` | 4.5 |
| `session3_speed_predictor.py` **GPU** | `results/session3_{speed,acceleration}_predictor.json` | 4.5, 5 |
| `run_repair_attribution.py` **GPU** | `results/p5_repair_attribution_L12.json` | 4.5 |
| `run_saddle_axis.py` **GPU** | `results/p5_saddle_axis_L{L}.json`, `results/p5_radial_steering_L{L}.json` | 4.5 |
| `run_straightening.py` | `results/p5_straightening.json` | 4.5 |
| `run_token_patching.py` **GPU** (`plan`, `forward`, `score`) | `results/p5_token_patching.json` | 4.6 |
| `run_object_vs_scene.py` | `results/p5_object_vs_scene_direction.json` | 4.6 |
| `run_motion_heads.py` **GPU** | `results/p5_motion_heads.json` | 4.7 |
| `run_makelov_ranking.py` **GPU** | `results/p5_makelov_ranking_L{L}.json` | 4.4 |

## Stimuli rendering (new clip sets, CPU)

| script | produces | REPORT § |
|---|---|---|
| `render_hard_stimuli.py` | `artifacts/stimuli/{paper_layout,hard}/`, `results/session2_stimuli_validation*.json` | 3.1, 4.5 |
| `make_stimulus_splits.py` | `splits/split_{set}.json` | 3.1 |
| `fix_stimuli_provenance.py` | corrects the split recorded in the stimulus-set step-1 results | 3.1 |
| `render_relational_stimuli.py` | `artifacts/stimuli/relational/`, `splits/split_relational.json`, `results/p5_relational_stimuli_validation.json` | 5 |
| `render_clock_stimuli.py` | time-rescaled clip sets under `artifacts/stimuli/` | 5 |

## Beyond the three variables (§5)

| script | produces | REPORT § |
|---|---|---|
| `run_object_permanence.py` | `results/p1a_object_permanence.json` | 5 |
| `run_support_cartesian_angle.py` | `results/p1a_support_cartesian_angle.json` | 5 |
| `run_speed_accel_angles.py` | `results/p5_speed_accel_angles.json` | 4.1, 5 |
| `run_accel_grid.py` **GPU** | `results/p5_accel_decorrelated_*.json` | 5 |
| `run_position_sheet.py` | `results/p2_sheet_{tag}.json` | 5 |
| `run_velocity_sheet.py`, `run_velocity_sheet_v2.py`, `fig_velocity_sheet_v2.py` | `results/p5_velocity_sheet{,_v2}.json`, `figures/fig_velocity_sheet_v2.png` | 5 |
| `run_velocity_sheet_predictor.py` **GPU** | `results/p5_velocity_sheet_predictor.json` | 5 |
| `run_position_predictor.py` **GPU** (`plan`, `forward`, `score`) | `results/session2_position_predictor.json` (start-position edit at blocks 12/22 through the predictor) | 4.5, 5 |
| `run_time_manifold.py`, `run_time_controls.py`, `run_time_positional.py` | `results/p5_time_manifold{,_controls}.json`, `results/p5_time_positional.json` | 5 |
| `run_time_shuffle.py` **GPU**, `time_static.py` **GPU** | `results/p5_time_shuffle.json`, `results/p5_time_static_control.json` | 5 |
| `time_predictor.py`, `time_patch_predictor.py` **GPU** | `results/p5_time_predictor.json`, `results/p5_time_patch_predictor.json` | 5 |
| `run_clock_test.py` **GPU** | `results/p5_clock_test_linearity.json` | 5 |
| `run_relational_motion.py`, `extract_relational_objpool.py` **GPU**, `run_relational_binding.py`, `run_relational_mlp.py` | `results/p5_relational_motion.json`, `figures/fig_relational_binding.png` | 5 |
| `run_relational_shape.py` | `results/p5_relational_shape.json` | 5 |
| `run_temporal_contact.py` **GPU**, `temporal_contact_score.py` | `results/p5_temporal_locality.json` | 5 |
| `run_contact_in_context.py` **GPU** | `results/p5_contact_in_context.json` | 5 |
| `run_contact_steer.py`, `run_contact_steer_heading_control.py` **GPU** | `results/p5_contact_steer.json`, `results/p5_contact_steer_heading_control.json` | 4.5, 5 |
| `contact_probe_domain.py` | `results/p5_contact_probe_domain.json` | 5 |
| `run_motion_geometry.py`, `run_motion_chord_leak.py`, `fig_motion_geometry.py` | `results/p5_motion_geometry.json` (+ `_parts/`), `figures/fig_motion_*.png` | 4.4 |

## Figures

| script | produces | REPORT § |
|---|---|---|
| `make_figures.py` | Part 1 figures in `figures/` from `results/*.json` (`--results`, `--figures` to redirect) | 3 |
| `../talk/make_talk_figs_v2.py` | the deck's figures in `talk/figures_v2/` | talk |
