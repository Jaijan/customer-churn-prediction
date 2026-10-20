# Two-day completion checklist

## Day 1 — Environment, data, and model

- [ ] Select a supported Python interpreter in VS Code and create `.venv`.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Confirm the IBM CSV is in `data/` and run `train_model.py`.
- [ ] Check generated metrics, class distribution, probability bounds, and confusion matrices in `models/metrics.json` and `models/test_predictions.csv`.
- [ ] Record only measured evaluation results in the report and slides.
- [ ] Launch the app and review each dashboard section.

## Day 2 — Dashboard, demo, and presentation

- [ ] Test the prediction form with several plausible and unusual category values.
- [ ] Check threshold behavior and explain false positives and false negatives.
- [ ] Rehearse a short walkthrough of EDA, model evaluation, and profile estimate.
- [ ] Prepare backup screenshots in case the live environment is unavailable.
- [ ] Verify slides match the saved evaluation run; retain limitations and no-causality language.

## Presentation talking points

- Why stratification preserves approximate churn proportions across splits.
- Why validation ROC-AUC is used for selection and the test set is held out.
- Why recall and precision matter alongside accuracy.
- How threshold, customer value, campaign cost, and outreach capacity affect decisions.
- Why a risk estimate is not a causal explanation or guaranteed retention outcome.
