# Presentation Outline — ChurnIQ

Use measured values from `models/metrics.json` only after training. No experiment metrics are currently available.

1. **Title and problem** — ChurnIQ: estimating customer churn risk from historical telecom data.
2. **Business motivation** — Prioritize retention review while accounting for value, outreach cost, and capacity.
3. **Dataset and features** — 7,043 IBM Telco customers; 5,174 retained and 1,869 churned; 20 model inputs after dropping customer ID; 11 blank TotalCharges values imputed.
4. **Exploratory analysis** — Churn rates: 42.7% month-to-month, 11.3% one-year, 2.8% two-year; median tenure 10 months among churned and 38 among retained. Associations only.
5. **Preprocessing and method** — Stratified 64% train, 16% validation, 20% test; imputation, scaling, one-hot encoding, balanced class weights.
6. **Model comparison** — Validation ROC-AUC: Logistic Regression 0.848; Random Forest 0.841. Logistic Regression selected.
7. **Evaluation and interpretation** — Test ROC-AUC: 0.841 vs 0.835; selected model precision 0.504, recall 0.783, F1 0.614, accuracy 0.738 at 0.50. Confusion matrix and threshold trade-offs; explain false positives and false negatives.
8. **Live dashboard demonstration** — Explore, inspect saved test results, change a hypothetical customer profile, adjust decision threshold.
9. **Business recommendations** — Prioritize review, investigate service context, test suitable offers and measure incremental outcomes.
10. **Limitations and conclusion** — Sample representativeness, drift, bias, association versus cause, and validation needed before operational use.
