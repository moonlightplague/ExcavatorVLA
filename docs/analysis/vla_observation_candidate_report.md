# VLA Observation Candidate Analysis

Episodes: 45
Causal rows: 10400
Provisional bucket capacity: 6377.600 particles
Screening sample rate: 2.000 Hz

## Time And Synchronization

- Raw dt p50: 579.332 ms
- Camera age p50/p95/p99: 7.017 / 334.690 / 897.737 ms
- Observation derivatives are causal; action labels use the next commanded position.

## Linear Grouped-Episode Screening

| Candidate | State dim | Mean NRMSE | Median NRMSE |
| --- | ---: | ---: | ---: |
| dynamics_memory_28d | 28 | 0.39024 | 0.37368 |
| replace_q_error_with_previous_action_28d | 28 | 0.38980 | 0.37679 |
| baseline_q_error_28d | 28 | 0.48978 | 0.49250 |
| replace_q_error_with_q_cmd_28d | 28 | 0.48530 | 0.49470 |
| retention_geometry_28d | 28 | 0.49258 | 0.49587 |
| replace_load_pose_with_pour_pose_28d | 28 | 0.49406 | 0.49791 |

This ridge result is a screening metric, not proof of final VLA rollout quality.
The final contract still requires nonlinear episode-level ablation and simulator rollout.

## Per-Phase Screening

| Candidate | pre_dig | approach_contact | insert_cut | pull_mid_cut | curl_to_hold_material | pull_exit_cut | secure_load | lift_carry | loaded_transit | unload_to_bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamics_memory_28d | 0.704 | 0.297 | 0.245 | 0.224 | 0.314 | 0.328 | 0.522 | 0.221 | 0.444 | 0.262 |
| replace_q_error_with_previous_action_28d | 0.695 | 0.302 | 0.230 | 0.224 | 0.319 | 0.323 | 0.536 | 0.222 | 0.462 | 0.260 |
| baseline_q_error_28d | 0.976 | 0.381 | 0.246 | 0.171 | 0.335 | 0.307 | 0.445 | 0.269 | 0.708 | 0.322 |
| replace_q_error_with_q_cmd_28d | 0.967 | 0.391 | 0.245 | 0.166 | 0.334 | 0.305 | 0.450 | 0.272 | 0.708 | 0.313 |
| retention_geometry_28d | 0.985 | 0.391 | 0.262 | 0.182 | 0.338 | 0.308 | 0.442 | 0.272 | 0.701 | 0.324 |
| replace_load_pose_with_pour_pose_28d | 0.987 | 0.384 | 0.248 | 0.173 | 0.336 | 0.305 | 0.443 | 0.269 | 0.707 | 0.325 |

## Baseline 28D

0. `swing`
1. `boom`
2. `arm`
3. `bucket`
4. `swing_velocity`
5. `boom_velocity`
6. `arm_velocity`
7. `bucket_velocity`
8. `swing_tracking_error`
9. `boom_tracking_error`
10. `arm_tracking_error`
11. `bucket_tracking_error`
12. `bucket_tip_base_x`
13. `bucket_tip_base_y`
14. `bucket_tip_height`
15. `bucket_load_base_x`
16. `bucket_load_base_y`
17. `bucket_load_height`
18. `dig_delta_upper_x`
19. `dig_delta_upper_y`
20. `dig_delta_z`
21. `unload_delta_upper_x`
22. `unload_delta_upper_y`
23. `unload_delta_z`
24. `truck_heading_upper_sin`
25. `truck_heading_upper_cos`
26. `bucket_fill_fraction`
27. `bucket_fill_rate_fraction_per_s`

## Effort Robust Statistics

- swing absolute effort p50/p95/p99/max: 17135.644 / 45925.065 / 252267.828 / 717027.750
- boom absolute effort p50/p95/p99/max: 64724.250 / 262750.048 / 1558316.845 / 10612393.000
- arm absolute effort p50/p95/p99/max: 6529.891 / 26011.782 / 680687.929 / 3454423.000
- bucket absolute effort p50/p95/p99/max: 0.763 / 15.750 / 151256.389 / 790357.188
