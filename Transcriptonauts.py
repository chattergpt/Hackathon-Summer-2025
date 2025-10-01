import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from lightgbm import LGBMRegressor

def single_profile(expr_collapsed: pd.DataFrame, gene: str):
    for k in (f"{gene}+ctrl", f"ctrl+{gene}"):
        if k in expr_collapsed.columns:
            return expr_collapsed[k].values
    return None

def load_and_prepare():
    train_data = pd.read_csv("train_set.csv")
    test_data = pd.read_csv("test_set.csv", header=None, names=["perturbation"])

    assert "Unnamed: 0" in train_data.columns, "Can't find gene identifier column"

    genes = train_data["Unnamed: 0"].astype(str).tolist()
    expr = train_data.drop(columns=["Unnamed: 0"]).copy()

    expr.columns = [c.split(".")[0] for c in expr.columns]
    expr_collapsed = expr.groupby(level=0, axis=1).mean()

    # baseline from single-perturbation columns
    singles = [c for c in expr_collapsed.columns if c.endswith("+ctrl") or c.startswith("ctrl+")]
    baseline = (
        expr_collapsed[singles].median(axis=1).values
        if len(singles) > 0
        else expr_collapsed.median(axis=1).values
    )

    return genes, expr_collapsed, test_data, baseline

# Building training data
def build_training(expr_collapsed: pd.DataFrame, genes: list[str]):
    print("Preparing training data...")
    X_rows, Y_rows = [], []
    skipped_genes = set()

    double_cols = [c for c in expr_collapsed.columns if "+ctrl" not in c and "ctrl+" not in c]

    for col in double_cols:
        if "+" not in col:
            continue
        g1, g2 = col.split("+", 1)

        profile1 = single_profile(expr_collapsed, g1)
        profile2 = single_profile(expr_collapsed, g2)

        if profile1 is None or profile2 is None:
            skipped_genes.update([g1, g2])
            continue

        feats = np.concatenate([profile1, profile2, profile1 * profile2, np.abs(profile1 - profile2), 0.5 * (profile1 + profile2)])
        X_rows.append(feats)
        Y_rows.append(expr_collapsed[col].values)

    X = np.array(X_rows)
    Y = np.array(Y_rows)

    features = (
        [f"A_{g}" for g in genes] +
        [f"B_{g}" for g in genes] +
        [f"A_B_{g}" for g in genes] +
        [f"Diff_{g}" for g in genes] +
        [f"Mean_{g}" for g in genes]
    )
    X_df = pd.DataFrame(X, columns=features)

    return X_df, Y, features, skipped_genes

def fit_models(X_df: pd.DataFrame, Y: np.ndarray, n_targets: int):
    models = []
    for i in tqdm(range(n_targets), desc="Model Training"):
        y = Y[:, i]
        reg = LGBMRegressor(random_state=7, verbosity=-1)
        reg.fit(X_df, y)
        models.append(reg)
    return models

def predict_test(models, genes, expr_collapsed, test_df, baseline, features):
    print("Predicting test set...")
    records = []
    used_baseline = set()

    for pert in test_df["perturbation"].astype(str):
        g1, g2 = pert.split("+", 1)

        profile1 = single_profile(expr_collapsed, g1)
        profile2 = single_profile(expr_collapsed, g2)

        if profile1 is None:
            used_baseline.add(g1)
            profile1 = baseline
        if v2 is None:
            used_baseline.add(g2)
            v2 = baseline

        x = np.concatenate([profile1, profile2, profile1 * profile2, np.abs(profile1 - profile2), 0.5 * (profile1 + profile2)]).reshape(1, -1)
        x_df = pd.DataFrame(x, columns=features)

        preds = [m.predict(x_df)[0] for m in models]
        records.extend((g, pert, float(p)) for g, p in zip(genes, preds))

    out_df = pd.DataFrame(records, columns=["gene", "perturbation", "expression"])
    return out_df, used_baseline

def main():
    genes, expr_collapsed, test_df, baseline = load_and_prepare()
    X_df, Y, features, skipped = build_training(expr_collapsed, genes)

    models = fit_models(X_df, Y, n_targets=len(genes))

    pred_df, used_baseline = predict_test(
        models=models,
        genes=genes,
        expr_collapsed=expr_collapsed,
        test_df=test_df,
        baseline=baseline,
        features=features,
    )

    # Set negative values to 0
    pred_df["expression"] = pd.to_numeric(pred_df["expression"], errors="coerce")
    pred_df["expression"] = np.maximum(pred_df["expression"], 0)
    pred_df.to_csv("prediction.csv", index=False)
    print(f"Done!")

if __name__ == "__main__":
    main()
