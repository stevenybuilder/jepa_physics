# Part 1 at the paper's literal 70/30 split (sensitivity / reproduction-of-protocol check)

Run: worktree at HEAD with WM_SPLIT_PATH=splits/split_paper70.json (stratified by value, seed 0, frame-hash grouped, 5 folds inside train), same code as the 80/20 results in results/. Finished 2026-09-27 20:13 ET. Figures in figures/split70/.


## Step 1 (CV R2 on train folds; onset = first point >= 90% of max)
| variable | split | n_train/n_test | pt1 | pt9 | peak (pt) | onset | final |
|---|---|---|---|---|---|---|---|
| direction | 80/20 | 1200/300 | 0.875 | 0.980 | 0.991 (22) | 2 [2, 2] | 0.990 |
| direction | 70/30 | 1050/450 | 0.868 | 0.980 | 0.990 (22) | 2 [2, 2] | 0.990 |
| speed | 80/20 | 1228/308 | 0.983 | 0.988 | 0.994 (19) | 1 [1, 1] | 0.993 |
| speed | 70/30 | 1075/461 | 0.983 | 0.987 | 0.994 (19) | 1 [1, 1] | 0.992 |
| acceleration | 80/20 | 1228/308 | 0.977 | 0.982 | 0.992 (21) | 1 [1, 1] | 0.990 |
| acceleration | 70/30 | 1075/461 | 0.977 | 0.982 | 0.993 (19) | 1 [1, 1] | 0.990 |
| (vx,vy) | 80/20 | 1228/308 | 0.985 | 0.984 | 0.990 (22) | 1 [1, 1] | 0.989 |
| (vx,vy) | 70/30 | 1075/461 | 0.984 | 0.982 | 0.989 (22) | 1 [1, 1] | 0.989 |
| (ax,ay) | 80/20 | 1228/308 | 0.975 | 0.980 | 0.989 (22) | 1 [1, 1] | 0.989 |
| (ax,ay) | 70/30 | 1075/461 | 0.973 | 0.978 | 0.989 (22) | 1 [1, 1] | 0.989 |

## Step 2 (INLP K at the paper layer, point 9, and at each variable's reference points)
| variable | point | split | nested K (fold range) | paper-protocol K | K at R2<0.3 |
|---|---|---|---|---|---|
| direction | 2 | 80/20 | 289 (197-383) | 395 | 152 / paper 259 |
| direction | 22 | 80/20 | 88 (69-97) | 94 | 37 / paper 39 |
| direction | 8 | 80/20 | 40 (33-47) | 83 | 21 / paper 42 |
| direction | 9 | 80/20 | 37 (33-42) | 46 | 23 / paper 25 |
| direction | 2 | 70/30 | 262 (234-280) | 382 | 131 / paper 255 |
| direction | 22 | 70/30 | 85 (69-98) | 95 | 33 / paper 33 |
| direction | 8 | 70/30 | 39 (31-41) | 34 | 20 / paper 20 |
| direction | 9 | 70/30 | 36 (28-43) | 35 | 21 / paper 22 |
| speed | 1 | 80/20 | 361 (350-385) | 410 | 361 / paper 410 |
| speed | 19 | 80/20 | 89 (78-100) | 103 | 89 / paper 103 |
| speed | 8 | 80/20 | 45 (39-53) | 45 | 45 / paper 45 |
| speed | 9 | 80/20 | 39 (36-45) | 45 | 39 / paper 45 |
| speed | 1 | 70/30 | 218 (196-296) | 380 | 255 / paper 380 |
| speed | 19 | 70/30 | 82 (73-93) | 98 | 82 / paper 98 |
| speed | 8 | 70/30 | 41 (37-44) | 48 | 41 / paper 48 |
| speed | 9 | 70/30 | 36 (34-43) | 40 | 36 / paper 40 |
| acceleration | 1 | 80/20 | 466 (336-509) | 554 | 493 / paper 554 |
| acceleration | 21 | 80/20 | 67 (61-70) | 74 | 67 / paper 74 |
| acceleration | 8 | 80/20 | 48 (44-51) | 50 | 48 / paper 50 |
| acceleration | 9 | 80/20 | 41 (36-44) | 47 | 41 / paper 47 |
| acceleration | 1 | 70/30 | 442 (322-543) | 520 | 476 / paper 520 |
| acceleration | 19 | 70/30 | 72 (46-84) | 88 | 82 / paper 88 |
| acceleration | 8 | 70/30 | 45 (38-49) | 49 | 45 / paper 49 |
| acceleration | 9 | 70/30 | 38 (35-40) | 47 | 38 / paper 47 |

## Step 3 (steering, paper protocol C.12; MAE-to-target at N probes, single target; eval probe out-of-fold R2)
| variable | point | split | K | N=1 | N=5 | N=10 | N=20 | N=K | MAE-to-true at K | eval probe OOF R2 | random null at N=5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direction | 2 | 80/20 | 289 | 85.71 | 77.34 | 64.99 | 21.37 | 6.94 | 85.47 | 0.860 | - |
| direction | 22 | 80/20 | 88 | 71.73 | 5.54 | 4.43 | 2.81 | 4.14 | 86.96 | 0.980 | - |
| direction | 9 | 80/20 | 37 | 78.38 | 8.73 | 3.12 | 2.92 | 2.69 | 86.99 | 0.967 | - |
| direction | 2 | 70/30 | 262 | 86.91 | 61.94 | 14.16 | 5.73 | 3.64 | 89.01 | 0.866 | - |
| direction | 22 | 70/30 | 85 | 62.93 | 7.31 | 3.36 | 2.71 | 3.16 | 89.22 | 0.983 | - |
| direction | 9 | 70/30 | 36 | 75.27 | 6.70 | 3.30 | 2.93 | 3.75 | 89.17 | 0.969 | - |
| speed | 1 | 80/20 | 361 | 0.81 | 0.23 | 0.09 | 0.09 | 0.06 | 0.95 | 0.972 | - |
| speed | 19 | 80/20 | 89 | 0.69 | 0.14 | 0.06 | 0.06 | 0.05 | 0.97 | 0.989 | - |
| speed | 9 | 80/20 | 39 | 0.69 | 0.16 | 0.08 | 0.08 | 0.08 | 0.95 | 0.978 | - |
| speed | 1 | 70/30 | 218 | 0.77 | 0.18 | 0.07 | 0.07 | 0.11 | 0.87 | 0.974 | - |
| speed | 19 | 70/30 | 82 | 0.52 | 0.06 | 0.06 | 0.06 | 0.06 | 0.94 | 0.990 | - |
| speed | 9 | 70/30 | 36 | 0.51 | 0.08 | 0.08 | 0.07 | 0.07 | 0.96 | 0.982 | - |
| acceleration | 1 | 80/20 | 466 | 2.18 | 0.75 | 0.30 | 0.24 | 0.23 | 2.47 | 0.954 | - |
| acceleration | 21 | 80/20 | 67 | 1.56 | 0.29 | 0.19 | 0.18 | 0.16 | 2.50 | 0.984 | - |
| acceleration | 9 | 80/20 | 41 | 1.77 | 0.40 | 0.23 | 0.23 | 0.23 | 2.44 | 0.968 | - |
| acceleration | 1 | 70/30 | 442 | 2.11 | 0.53 | 0.27 | 0.31 | 0.23 | 2.42 | 0.964 | - |
| acceleration | 19 | 70/30 | 72 | 1.39 | 0.21 | 0.18 | 0.19 | 0.17 | 2.46 | 0.989 | - |
| acceleration | 9 | 70/30 | 38 | 1.60 | 0.25 | 0.20 | 0.21 | 0.19 | 2.44 | 0.974 | - |

Reading: every qualitative finding is unchanged (onsets identical; K in the tens at point 9 for all three variables; steering shape: 1 probe fails, 5–10 probes reach the target, MAE-to-true rises to ~90° as MAE-to-target falls). The paper-protocol K for direction at point 9 is 35 at 70/30 vs 46 at 80/20 because that count reads the test split at every round; the nested K (36 vs 37) is stable. Split of record for the report stays split_v1 (80/20) so Part 1 and Part 2 read the same clips; this table is the check that the paper's ratio changes nothing.
