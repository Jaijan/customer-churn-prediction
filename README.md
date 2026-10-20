# ChurnIQ — Customer Retention Intelligence

A locally runnable customer churn case study using the IBM Telco Customer Churn sample, scikit-learn, and Streamlit. It validates the source data, compares interpretable Logistic Regression with Random Forest, and provides exploratory charts, held-out evaluation, and an individual profile estimate.

## Business problem

Telecom teams may use historical customer information to prioritize retention reviews. Churn prediction estimates risk; it does not identify causes or guarantee that an offer will prevent a customer leaving. Campaign decisions should also consider customer value, costs, capacity, and measured intervention outcomes.

## Dataset

Source: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The local project includes the public IBM source file at `data/Telco-Customer-Churn.csv`. The loader checks `Churn`, validates its Yes/No values, converts `TotalCharges` to numeric, removes exact duplicate rows, and excludes `customerID` from model features. Missing values are imputed inside the model pipeline.

## Windows and VS Code setup

Open this folder in VS Code. In the integrated PowerShell terminal, create a local environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code interpreter if needed. Activation is optional. If `py` is not installed, use a supported Python interpreter's full executable path in place of `py`. This prepared workspace includes Python 3.13.16 at `.tools\python313\python.exe`; use it to create `.venv` if the Windows launcher is unavailable. `.tools/` is local and Git-ignored.

## Train and launch

From the project root:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\streamlit.exe run app.py
```

An alternate data path can be supplied with `--data path\to\file.csv`. Artifacts are written to `models/`: the selected fitted pipeline, both candidate pipelines for interpretation, `metrics.json`, and real held-out `test_predictions.csv`. Model outputs are ignored by Git and can be regenerated.

## Modeling and evaluation

The script reserves 20% as an untouched final test partition, with a 16% validation partition of the remaining development data. All splits are stratified with seed 42. Missing numeric values use median imputation and scaling; categorical values use most-frequent imputation and one-hot encoding with unknown-category support. Logistic Regression and Random Forest use class weights to account for the observed class imbalance. Candidate selection uses validation ROC-AUC, then each candidate is refit on development data and scored on the test set. Saved classification metrics use a 0.50 threshold; test labels are not used to choose the model or threshold.

## Dashboard

- Executive KPIs and observed churn patterns
- EDA for churn, contract, tenure, charges, and selectable categories
- Candidate comparison, saved test confusion matrix, ROC and precision-recall curves
- Single-profile probability estimate using the same persisted preprocessing pipeline
- Business interpretation and limitations

The interactive threshold is configurable for risk categorization. Reported candidate test metrics remain at the documented 0.50 threshold.

## Limitations

This is historical public sample data for a fictional telecom provider. Sampling, missingness, measurement choices, and time drift limit generalization. Associations and feature importance are not causal evidence. Before operational use, validate on current data, assess calibration and subgroup performance, and test retention actions with measured outcomes.
