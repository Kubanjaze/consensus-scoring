# consensus-scoring — Phase 43

Average-rank consensus scoring combining RF, LightGBM, and SVM classifiers.
Evaluates whether combining models improves ROC-AUC and EF@10% over individual models.

## Usage

```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv --threshold 7.0
```
