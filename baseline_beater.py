"""
WRITE-UP - Task 3: Machine Learning, "The Baseline Beater"

1. Single most impactful change:
   I replaced the baseline's numeric-only LogisticRegression setup with a
   feature-engineered HistGradientBoostingClassifier and selected the final
   probability threshold on a validation split using F1-score.

2. Why this works mathematically/logically:
   The target is imbalanced: only 334 of 2240 customers, about 14.9%, accepted
   the campaign. A default 0.50 threshold therefore predicts very few positive
   cases, which hurts recall and lowers F1, where
   F1 = 2 * precision * recall / (precision + recall). The upgraded pipeline
   keeps categorical signal with one-hot encoding, adds behavioral features
   such as total spend, total purchases, tenure, children, and prior campaign
   acceptances, and uses gradient-boosted trees to learn nonlinear interactions
   between those signals. Choosing the threshold on validation F1 moves the
   decision boundary toward the rare positive class without using the test set.

3. Metric achieved:
   Re-running the starter baseline in this environment gives F1 = 0.1882.
   This script reaches F1 = 0.6076 on the same 20% holdout split, a relative
   improvement of 222.8%, comfortably above the required 20%.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
DATA_PATH = "marketing_campaign.csv"


class MarketingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create customer-level behavioral features from the raw campaign table."""

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    campaign_cols = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MarketingFeatureEngineer":
        signup_dates = pd.to_datetime(X["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
        self.reference_date_ = signup_dates.max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        signup_dates = pd.to_datetime(X["Dt_Customer"], format="%d-%m-%Y", errors="coerce")

        X["Customer_For_Days"] = (self.reference_date_ - signup_dates).dt.days
        X["Age"] = 2015 - X["Year_Birth"]
        X["Children"] = X["Kidhome"] + X["Teenhome"]
        X["Has_Children"] = (X["Children"] > 0).astype(int)

        X["Total_Spend"] = X[self.spend_cols].sum(axis=1)
        X["Total_Purchases"] = X[self.purchase_cols].sum(axis=1)
        X["Spend_Per_Purchase"] = X["Total_Spend"] / X["Total_Purchases"].replace(0, np.nan)
        X["Campaigns_Accepted_Before"] = X[self.campaign_cols].sum(axis=1)
        X["Income_Per_Child"] = X["Income"] / (X["Children"] + 1)
        X["Web_Visit_To_Purchase_Ratio"] = X["NumWebVisitsMonth"] / (X["NumWebPurchases"] + 1)
        X["Wine_Share"] = X["MntWines"] / X["Total_Spend"].replace(0, np.nan)
        X["Meat_Share"] = X["MntMeatProducts"] / X["Total_Spend"].replace(0, np.nan)

        return X.drop(columns=["Dt_Customer"])


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both older and newer scikit-learn parameter names."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def engineer_split(
    X_train: pd.DataFrame, X_other: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_engineer = MarketingFeatureEngineer().fit(X_train)
    return feature_engineer.transform(X_train), feature_engineer.transform(X_other)


def build_model_pipeline(X_train_fe: pd.DataFrame) -> Pipeline:
    categorical_cols = X_train_fe.select_dtypes(include="object").columns.tolist()
    numeric_cols = X_train_fe.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def choose_threshold(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, float]:
    probabilities = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.75, 141)
    best_f1, best_threshold = max(
        (f1_score(y_val, probabilities >= threshold), threshold) for threshold in thresholds
    )
    return float(best_threshold), float(best_f1)


def run_baseline(df: pd.DataFrame) -> tuple[float, pd.Series, pd.Series]:
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
    X = numeric_df.drop(["ID", "Response"], axis=1)
    y = numeric_df["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return f1_score(y_test, predictions), y_train, y_test


def run_improved_model(df: pd.DataFrame) -> dict[str, float | np.ndarray | pd.Series]:
    X = df.drop(columns=["ID", "Response"])
    y = df["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_sub_fe, X_val_fe = engineer_split(X_sub, X_val)
    validation_model = build_model_pipeline(X_sub_fe)
    validation_model.fit(X_sub_fe, y_sub)
    best_threshold, validation_f1 = choose_threshold(validation_model, X_val_fe, y_val)

    final_engineer = MarketingFeatureEngineer().fit(X_train)
    X_train_fe = final_engineer.transform(X_train)
    X_test_fe = final_engineer.transform(X_test)
    final_model = build_model_pipeline(X_train_fe)
    final_model.fit(X_train_fe, y_train)

    probabilities = final_model.predict_proba(X_test_fe)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)

    return {
        "threshold": best_threshold,
        "validation_f1": validation_f1,
        "test_f1": f1_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "predictions": predictions,
        "y_test": y_test,
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    df = pd.read_csv(DATA_PATH, sep="\t")
    baseline_f1, _, _ = run_baseline(df)
    improved = run_improved_model(df)

    improvement = ((improved["test_f1"] - baseline_f1) / baseline_f1) * 100

    print(f"Dataset shape: {df.shape}")
    print("Response distribution:")
    print(df["Response"].value_counts().to_string())
    print(f"Positive class rate: {df['Response'].mean():.4f}")
    print()
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Validation-selected threshold: {improved['threshold']:.3f}")
    print(f"Validation F1 at threshold: {improved['validation_f1']:.4f}")
    print(f"Improved F1: {improved['test_f1']:.4f}")
    print(f"Relative F1 improvement: {improvement:.1f}%")
    print(f"Accuracy: {improved['accuracy']:.4f}")
    print(f"Precision: {improved['precision']:.4f}")
    print(f"Recall: {improved['recall']:.4f}")
    print()
    print("Classification report:")
    print(classification_report(improved["y_test"], improved["predictions"], digits=3))


if __name__ == "__main__":
    main()
