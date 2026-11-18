# PathSnap – Technical Report (Harsh, Hardhick, Debasditya, Pranav)

## 1. Data and preprocessing
80 training trips (207,747 GPS points) and 10 student-test trips (19,158 points); each trip ships its own road network
(2k–166k arcs, directed, ~99.7% have a reverse arc). Cleaning: drop non-monotonic timestamps and jumps >60 m/s, then a
3-point rolling median. Features use a per-trip local metric projection. Candidates: all arcs within 50 m via a
cKDTree over points sampled every 10 m along each arc, at most 12 per point (150 m fallback if none). Candidate miss rate
is 0.4% (train) and 0.03% (student test).

**Labels.** The reference route is aligned to the track with a monotone dynamic program (positions may only stay or
advance); points farther than 40 m from every route arc stay unlabelled. Two training trips (00000033, 00000054) have
reference routes that lie >100 m from their GPS in most points (labelled fraction <50%); they were excluded as
noisy-label trips. 78 trips remain (1.5M candidate rows before sampling).

**Features (19):** distance, distance relative to the nearest candidate, rank, candidate count, projection fraction,
arc length, heading difference (short and long window) + cosine, heading validity, speed, signed lateral offset,
distance of points t±2 to the same arc, along-arc motion, out/in degree, has-reverse-arc.

## 2. Model comparison (train only, GroupKFold by trip, 5 folds)
Seed 42; 120 sampled points per trip. Decision rule for macro-F1/top-1: highest-scoring candidate per GPS point.
Nearest-arc baseline (no learning) top-1 = 59.8% on the same CV sample.

| Model | Family | CV Top-1 | CV macro F1 | Precision | Recall | F1 | ROC-AUC | Log loss |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | linear | 0.8906 | 0.9383 | 0.867 | 0.847 | 0.857 | 0.9845 | 0.0994 |
| K-Nearest Neighbors | instance | 0.8826 | 0.9337 | 0.883 | 0.839 | 0.860 | 0.9830 | 0.1072 |
| Random Forest | bagging | 0.9045 | 0.9463 | 0.917 | 0.878 | 0.897 | 0.9921 | 0.0683 |
| **Hist. Gradient Boosting** | boosting | **0.9070** | **0.9477** | 0.908 | 0.893 | 0.901 | 0.9928 | 0.0676 |
| MLP | neural | 0.8964 | 0.9416 | 0.897 | 0.869 | 0.883 | 0.9896 | 0.0798 |

Precision/recall/F1 are positive-class at threshold 0.5. Fold std of top-1 is 1.2–1.5 points, so RF vs HGB is within noise;
HGB was selected by the pre-declared rule (CV top-1, tie-break log loss).

## 3. Bayesian optimisation
Optuna TPE (20 trials, seed 42, 4 group folds by trip, 100 points/trip) over iterations, learning rate, leaf nodes,
min leaf size, L2, depth. Best tuning CV macro-F1 0.9511. **Caveat:** the tuning sample differs from the comparison
sample, so 0.9511 vs 0.9477 is not a like-for-like gain; on student test the tuned and default models are equal
within noise (Top-1 0.9102 vs 0.9107). The tuned model is deployed.

## 4. HMM / Viterbi
States = candidate arcs. Emission = alpha·log P_clf(normalised per point) − (1−alpha)·d²/2σ². Transition =
−|d_network − d_GPS|/β(dt), d_network from bounded Dijkstra on the directed graph; unreachable −25..−35 (flags a break);
U-turn penalty; hard restart on gaps >180 s or >3 km. Decoded arcs are joined by shortest paths into one connected route.
HMM parameters (alpha 0.23, σ 26 m, β0 7.6 m, β/s 2.2, invalid 35, U-turn 10.9) were tuned with Optuna on
**out-of-fold** scores of 30 train trips: top-1 0.920 (classifier) -> 0.949, route F1 0.980.

## 5. Final student-test results (first and only use; 10 trips, 19,158 points)
| Model (trained on train only) | Top-1 | Macro F1 | ROC-AUC | Log loss | Miss rate | s / 1k points |
|---|---|---|---|---|---|---|
| Nearest-arc baseline | 0.5565 | 0.7452 | – | – | 0.0003 | – |
| Logistic Regression | 0.8779 | 0.9299 | 0.9862 | 0.0969 | 0.0003 | 0.001 |
| KNN | 0.8789 | 0.9305 | 0.9829 | 0.1083 | 0.0003 | 1.29 |
| Random Forest | 0.9046 | 0.9452 | 0.9933 | 0.0654 | 0.0003 | 0.030 |
| Hist. Gradient Boosting | 0.9107 | 0.9488 | 0.9937 | 0.0640 | 0.0003 | 0.057 |
| MLP | 0.9031 | 0.9444 | 0.9920 | 0.0732 | 0.0003 | 0.008 |
| HGB (Bayes-tuned, deployed) | 0.9102 | 0.9485 | 0.9938 | 0.0637 | 0.0003 | 0.034 |

**Route reconstruction (deployed model, full pipeline, mean over trips):**
| | Classifier only | Classifier + HMM |
|---|---|---|
| Point Top-1 (point-weighted) | 0.910 | **0.979** |
| Route F1 vs reference (length-weighted arcs) | 0.847 | **0.992** |
| Route precision / recall | 0.746 / 0.998 | 0.986 / 0.997 |
| Route continuity (raw decoded arcs) | 0.823 | 0.919 |
| Route continuity (after gap bridging) | – | 0.999 |

Per-trip results: `results/student_test_routes_per_trip.csv`; Folium maps: `results/maps/`.
End-to-end time is 0.1–0.8 s per trip.

## 6. Limitations / honest notes
- Labels come from a DP alignment of the reference route, not human annotation; two noisy-label training trips were dropped.
- Classifier-only continuity counts consecutive distinct arcs that touch; "after gap bridging" inserts shortest-path arcs, so
  it is high by construction and mainly shows the HMM leaves few unbridgeable breaks (0.1%).
- Route precision uses an arc-set comparison, so it ignores traversal order.
- Training used 300 sampled points per trip for speed, not every point.
- Random Forest and HGB are statistically indistinguishable in CV; HGB was chosen by the declared rule.
