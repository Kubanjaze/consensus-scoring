# Phase 43 — Consensus Scoring (Average Rank)

**Version:** 1.1 | **Tier:** Micro | **Date:** 2026-03-26

## Goal
Combine RF, LGBM, and SVM scores via average rank consensus.
Compare individual vs consensus ROC-AUC and EF@10%.

CLI: `python main.py --input data/compounds.csv --threshold 7.0`

Outputs: consensus_scores.csv, consensus_comparison.png

## Logic
- Train 3 models (RF, LGB, SVM-RBF) with LOO-CV, get probability scores
- Rank each model's scores (1=highest)
- Consensus score = mean(rank_rf, rank_lgbm, rank_svm) — lower rank = better
- Convert to score: score = N - mean_rank + 1 (higher = better)
- Compare ROC-AUC and EF@10% for each model + consensus

## Actual Results (v1.1)

| Model | ROC-AUC | PR-AUC | EF@10% |
|---|---|---|---|
| RF | 0.8267 | 0.9101 | 1.50× |
| LightGBM | 0.4422 | 0.7157 | 0.75× |
| SVM | 0.7889 | 0.8897 | 1.50× |
| Consensus | 0.2600 | 0.5426 | 0.00× |

**Key insight:** Average-rank consensus is hurt by LightGBM (ROC-AUC=0.44, below random). RF and SVM individually achieve 0.83/0.79. Consensus only works when all component models are at least better than random — one bad model contaminates the ensemble. RF is the best single model.
