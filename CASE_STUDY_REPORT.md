# Customer Churn Prediction — Case Study Report

## Abstract

This project builds a reproducible pipeline to estimate customer churn from the IBM Telco Customer Churn sample. Logistic Regression and Random Forest are compared using validation ROC-AUC, then assessed on a separate held-out test set. A Streamlit dashboard presents source-data summaries, evaluation, and individual profile estimates. Measured model results are recorded here only after a successful local training run.

## Introduction and business problem

Customer churn can affect recurring revenue and service planning. A risk score may help a retention team prioritize customer reviews, but it is not a causal explanation and does not show that a campaign will prevent churn.

## Objectives

- Validate the target and source fields and exclude customer identifiers.
- Build a preprocessing pipeline that learns transformations only from training data.
- Compare a linear baseline and a nonlinear classifier.
- Keep model selection separate from final test evaluation.
- Present genuine data and model outputs in a local dashboard.

## Dataset

The project uses IBM's public Telco Customer Churn data, commonly distributed through Kaggle. The local source file is `data/Telco-Customer-Churn.csv`. It contains customer service, account, and billing attributes and a binary `Churn` outcome. `customerID` is removed; `TotalCharges` is converted to numeric. Exact duplicate rows are removed. Missing counts and class distribution are captured in the training metadata.

**Observed dataset summary:** 7,043 rows and 20 model input features after excluding `customerID`; 11 blank `TotalCharges` values were converted to missing numeric values. The target distribution is 5,174 retained (73.46%) and 1,869 churned (26.54%). No exact full-row duplicates were found in the source. Missing values are imputed during fitting.

## Methodology and preprocessing

The data is split into a stratified 80% development set and 20% final test set (seed 42). A stratified validation split reserves 20% of development for model selection. Numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding with unknown-category handling. The identifier is excluded. Class weighting is applied to both models.

## Algorithms considered

- **Logistic Regression:** interpretable linear baseline with balanced class weights.
- **Random Forest:** nonlinear ensemble with balanced class weights and fixed random seed.

## Evaluation methodology

Validation ROC-AUC selects the candidate. The selected pipeline is trained on all development rows; both candidates are independently refit on development before final test scoring. The documented threshold is 0.50. Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report are saved for both candidates. No threshold is selected using final test labels.

## Experimental results

Training completed with Python 3.13.16, pandas 2.3.3, NumPy 2.5.4, scikit-learn 1.9.1, and seed 42. Validation ROC-AUC was 0.848 for Logistic Regression and 0.841 for Random Forest, so Logistic Regression was selected before final testing. The held-out test set contained 1,409 customers (374 churned).

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.841 | 0.738 | 0.504 | 0.783 | 0.614 |
| Random Forest | 0.835 | 0.760 | 0.536 | 0.714 | 0.612 |

At the fixed 0.50 threshold, Logistic Regression's confusion matrix (actual rows No/Yes; predicted columns No/Yes) was `[[747, 288], [81, 293]]`. It caught 293 of 374 churners, while flagging 288 retained customers. Random Forest's corresponding matrix was `[[804, 231], [107, 267]]`. These illustrate the outreach trade-off: Logistic Regression found more churners, while the Random Forest produced fewer false alarms. These metrics do not establish business benefit.

## Interpretation and business recommendations

In the observed dataset, churn prevalence was 42.7% for month-to-month contracts, 11.3% for one-year contracts, and 2.8% for two-year contracts. Median tenure was 10 months among churned customers and 38 among retained customers. Median monthly charges were 79.65 and 64.43 respectively (dataset units). These are descriptive associations. The selected Logistic Regression's largest positive transformed coefficients included fiber optic service and month-to-month contract; tenure and two-year contract had strong negative coefficients. These conditional model relationships are not causal effects and depend on the encoded predictors and fitted sample.

For a future deployment, treat model output as one input into a prioritized review queue. Balance probability against customer value, contact costs, retention capacity, and customer preferences. Evaluate offers through controlled experiments and track retention, cost, and customer experience outcomes.

## Limitations and future work

The data is a historical public sample that may not represent current populations or current service conditions. Random splitting does not test future-period generalization. Missing values, sampling and measurement decisions may introduce bias. Predictive association is not causation. Future work should validate on time-based splits, measure calibration and subgroup performance, evaluate threshold trade-offs against costs, and test interventions experimentally.
