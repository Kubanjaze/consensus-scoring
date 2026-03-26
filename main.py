import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
RDLogger.DisableLog("rdApp.*")

FAMILY_COLORS = {"benz": "#4C72B0", "naph": "#DD8452", "ind": "#55A868",
                 "quin": "#C44E52", "pyr": "#8172B2", "bzim": "#937860", "other": "#808080"}

def load_compounds(path, threshold):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None: n_bad += 1; continue
        try:
            pic50 = float(row["pic50"])
        except (KeyError, ValueError):
            continue
        if np.isnan(pic50): continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        fam = str(row["compound_name"]).split("_")[0]
        records.append({"compound_name": str(row["compound_name"]),
                        "family": fam if fam in FAMILY_COLORS else "other",
                        "active": int(pic50 >= threshold), "fp": list(fp)})
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)

def loo_scores(model_fn, X, y):
    loo = LeaveOneOut()
    scores = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[0, 1]
    return scores

def compute_ef(y_true, y_score, k_frac):
    n = len(y_true); k = max(1, int(np.round(n * k_frac)))
    total_hits = y_true.sum()
    if total_hits == 0: return 0.0
    order = np.argsort(y_score)[::-1]
    hits_topk = y_true[order[:k]].sum()
    return round(float((hits_topk / k) / (total_hits / n)), 3)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input, args.threshold)
    X = np.array(df["fp"].tolist(), dtype=float)
    y = df["active"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running LOO-CV for RF...")
    rf_scores = loo_scores(lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X, y)
    print("Running LOO-CV for LightGBM...")
    lgbm_scores = loo_scores(lambda: LGBMClassifier(n_estimators=100, num_leaves=15, random_state=42, verbose=-1), X, y)
    print("Running LOO-CV for SVM...")
    svm_scores = loo_scores(lambda: SVC(kernel="rbf", probability=True, random_state=42, C=1.0), X_scaled, y)

    n = len(y)
    # Rank each (rank 1 = highest score)
    def score_to_rank(s):
        return n + 1 - np.argsort(np.argsort(s)[::-1]) - 1

    rank_rf = score_to_rank(rf_scores)
    rank_lgbm = score_to_rank(lgbm_scores)
    rank_svm = score_to_rank(svm_scores)
    consensus_rank = (rank_rf + rank_lgbm + rank_svm) / 3
    consensus_score = n - consensus_rank + 1  # invert: higher = better

    model_scores = {
        "RF": rf_scores, "LightGBM": lgbm_scores,
        "SVM": svm_scores, "Consensus": consensus_score,
    }

    results = {}
    for name, s in model_scores.items():
        results[name] = {"roc_auc": round(roc_auc_score(y, s), 4),
                         "pr_auc": round(average_precision_score(y, s), 4),
                         "ef_10": compute_ef(y, s, 0.10)}

    # Save scores
    score_df = pd.DataFrame({
        "compound_name": df["compound_name"], "family": df["family"],
        "y_true": y, "rf_score": rf_scores, "lgbm_score": lgbm_scores,
        "svm_score": svm_scores, "consensus_score": consensus_score,
    })
    score_df.to_csv(os.path.join(args.output_dir, "consensus_scores.csv"), index=False)
    print(f"Saved: {args.output_dir}/consensus_scores.csv")

    # Plot comparison
    names = list(results.keys())
    roc_aucs = [results[n]["roc_auc"] for n in names]
    ef10s = [results[n]["ef_10"] for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    bars1 = ax1.bar(names, roc_aucs, color=colors, edgecolor="white")
    ax1.axhline(0.5, color="k", linestyle="--", lw=1)
    for b, v in zip(bars1, roc_aucs):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylabel("ROC-AUC"); ax1.set_title("ROC-AUC Comparison", fontweight="bold")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(names, ef10s, color=colors, edgecolor="white")
    ax2.axhline(1.0, color="k", linestyle="--", lw=1)
    for b, v in zip(bars2, ef10s):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.2f}x", ha="center", fontsize=9)
    ax2.set_ylabel("EF@10%"); ax2.set_title("EF@10% Comparison", fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    plt.suptitle("Consensus Scoring vs Individual Models (LOO-CV)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "consensus_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output_dir}/consensus_comparison.png")

    print(f"\n--- Model Comparison (threshold={args.threshold}) ---")
    print(f"  {'Model':<12}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'EF@10%':>7}")
    for name in names:
        r = results[name]
        print(f"  {name:<12}  {r['roc_auc']:>8.4f}  {r['pr_auc']:>8.4f}  {r['ef_10']:>7.2f}x")
    print("\nDone.")

if __name__ == "__main__":
    main()
