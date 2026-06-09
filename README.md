# Task 3: Machine Learning - The Baseline Beater

This repository improves the starter model for predicting the `Response` column in
`marketing_campaign.csv`.

## What Changed

The starter notebook used only numeric columns, filled missing values with `0`,
and trained a basic `LogisticRegression` model. That approach missed categorical
signal, ignored useful behavioral combinations, and predicted too few positive
campaign responses because the target class is imbalanced.

The improved solution is in `baseline_beater.py`. It:

- Preserves categorical features with one-hot encoding.
- Adds customer behavior features such as age, customer tenure, total spend,
  total purchases, spend per purchase, children, income per child, and previous
  campaign acceptances.
- Uses `HistGradientBoostingClassifier` to model nonlinear feature interactions.
- Selects the probability threshold on a validation split to maximize F1-score.

## Results

Using the same `random_state=42` 20% holdout split:

| Model | F1-score |
| --- | ---: |
| Starter baseline rerun | 0.1882 |
| Improved model | 0.6076 |

Relative F1 improvement: **222.8%**

The starter notebook's saved output may show a different baseline value because
of environment/version differences, but the improved model remains comfortably
above the required 20% gain.

## How to Run

```bash
pip install -r requirements.txt
python baseline_beater.py
```

Expected key output:

```text
Baseline F1: 0.1882
Validation-selected threshold: 0.140
Validation F1 at threshold: 0.6269
Improved F1: 0.6076
Relative F1 improvement: 222.8%
```

## Mandatory Write-up

The required 3-point write-up is included at the very top of
`baseline_beater.py` as a module docstring.
