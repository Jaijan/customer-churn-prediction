"""Train and evaluate reproducible Telco churn classifiers."""
import argparse
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = [ROOT / "data" / n for n in (
    "WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv")]
THRESHOLD = 0.5

def find_data(path=None):
    if path:
        p = Path(path)
        if not p.is_absolute(): p = ROOT / p
        if not p.is_file(): raise FileNotFoundError(f"Dataset not found: {p}")
        return p
    for p in DEFAULT_CANDIDATES:
        if p.is_file(): return p
    raise FileNotFoundError("Telco CSV not found in data/. Place the IBM CSV there or pass --data PATH.")

def load_data(path=None):
    df = pd.read_csv(find_data(path))
    df.columns = [str(c).strip() for c in df.columns]
    if "Churn" not in df: raise ValueError("Dataset must contain a Churn target column.")
    if "customerID" in df and df["customerID"].duplicated().any():
        raise ValueError("customerID values must be unique.")
    df = df.drop_duplicates().copy()
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    target = df["Churn"].astype(str).str.strip()
    invalid = sorted(set(target) - {"Yes", "No"})
    if invalid: raise ValueError(f"Unexpected Churn values: {invalid}")
    df["Churn"] = target.map({"No": 0, "Yes": 1}).astype("int8")
    df = df.drop(columns=["customerID"], errors="ignore")
    if df.empty or df["Churn"].nunique() != 2: raise ValueError("Both Churn classes must be present.")
    return df

def build_pipeline(X, model):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])

def evaluate(pipe, X, y):
    probs = pipe.predict_proba(X)[:, 1]
    pred = (probs >= THRESHOLD).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, labels=[0, 1],
            target_names=["No churn", "Churn"], output_dict=True, zero_division=0)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to Telco CSV")
    args = parser.parse_args()
    df = load_data(args.data)
    y, X = df["Churn"], df.drop(columns="Churn")
    # Reserve a final stratified test set. Choose model using validation ROC-AUC only.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.2, random_state=42, stratify=y_dev)
    factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1),
    }
    validation, candidates = {}, {}
    for name, factory in factories.items():
        pipe = build_pipeline(X_train, factory()).fit(X_train, y_train)
        validation[name] = float(roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]))
        candidates[name] = factory
        print(f"{name}: validation ROC-AUC={validation[name]:.3f}")
    winner = max(validation, key=validation.get)
    results = {}
    # Refit each candidate on all development data for fair held-out comparison.
    fitted = {}
    for name, factory in candidates.items():
        model = build_pipeline(X_dev, factory()).fit(X_dev, y_dev)
        fitted[name] = model
        results[name] = evaluate(model, X_test, y_test)
        m = results[name]
        print(f"{name}: test ROC-AUC={m['roc_auc']:.3f}, F1={m['f1']:.3f}, recall={m['recall']:.3f}, precision={m['precision']:.3f}")
    final = fitted[winner]
    run_id = uuid.uuid4().hex
    for model in fitted.values(): model.evaluation_run_id = run_id
    out = ROOT / "models"; out.mkdir(exist_ok=True)
    joblib.dump(final, out / "churn_pipeline.joblib")
    joblib.dump(fitted, out / "candidate_pipelines.joblib")
    meta = {"evaluation_run_id": run_id, "selected_model": winner, "selection_metric": "validation ROC-AUC",
        "validation_roc_auc": validation, "metrics": results, "feature_columns": X.columns.tolist(),
        "target": "Churn", "threshold": THRESHOLD, "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]), "missing_values": {k: int(v) for k,v in df.isna().sum().items()},
        "class_distribution": {"No": int((y == 0).sum()), "Yes": int((y == 1).sum())},
        "churn_rate": float(y.mean()), "test_size": .2, "validation_size_of_development": .2,
        "random_state": 42, "duplicate_rows_removed": "exact full-row duplicates only"}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pred = final.predict_proba(X_test)[:, 1]
    test_out = X_test.copy(); test_out["actual_churn"] = y_test.values
    test_out["churn_probability"] = pred; test_out["predicted_churn"] = (pred >= THRESHOLD).astype(int)
    test_out["evaluation_run_id"] = run_id
    test_out.to_csv(out / "test_predictions.csv", index=False)
    print(f"Selected by validation ROC-AUC: {winner}; saved artifacts in {out}")

if __name__ == "__main__": main()
