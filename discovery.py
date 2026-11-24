"""Compare a low-code workflow (PyCaret) with a manual one (scikit-learn).

Business problem: a Portuguese bank runs phone campaigns to sell term deposits.
Before an agent dials, predict whether the client will subscribe ("yes"/"no")
so the call list can be prioritised toward likely subscribers.

Run top to bottom with:  uv run discovery.py
Outputs:
    images/pycaret_comparison.csv / .html  - PyCaret compare_models() table
    images/Confusion Matrix.png           - PyCaret confusion matrix (best model)
    model.joblib                          - fitted scikit-learn Pipeline
"""

import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # quiet joblib warning on Windows

import matplotlib

matplotlib.use("Agg")  # save plots to file, never open a window

import joblib
import pandas as pd

DATA_PATH = Path("data/bank.csv")
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

TARGET = "y"
# duration = length of the call in seconds. It is only known AFTER the call
# ends (and a 0-second call is always "no"), so using it would leak the answer.
# The dataset has no ID column, so nothing else needs dropping.
DROP_COLUMNS = ["duration"]

df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLUMNS)
# Encode the target as 1 = "yes" (subscribes), 0 = "no". With the raw strings
# PyCaret reported *weighted* Recall/Precision/F1 averaged over both classes,
# which hides how well we find the rare "yes" class.
df[TARGET] = (df[TARGET] == "yes").astype(int)
print(f"Data: {df.shape[0]} rows, {df.shape[1] - 1} features")
print(df[TARGET].value_counts(normalize=True).round(3).to_string(), "\n")


# =============================================================================
# Part A: PyCaret workflow
# =============================================================================
pycaret_start = time.perf_counter()

from pycaret.classification import compare_models, plot_model, pull, setup

setup(data=df, target=TARGET, session_id=42, verbose=False)

# Rank by F1: the classes are imbalanced (~88% "no"), so accuracy would reward
# a model that never predicts "yes". F1 balances finding subscribers (recall)
# against wasting calls on non-subscribers (precision).
top3 = compare_models(n_select=3, sort="F1", verbose=False)
comparison = pull()
comparison.to_csv(IMAGES / "pycaret_comparison.csv")
comparison.to_html(IMAGES / "pycaret_comparison.html", float_format="%.4f")
print("PyCaret comparison (10-fold CV on the 70% training split):")
print(comparison.to_string(), "\n")

best = top3[0]
print(f"PyCaret best model: {type(best).__name__}")

# save=str saves "Confusion Matrix.png" into that folder instead of showing it
plot_model(best, plot="confusion_matrix", save=str(IMAGES))

pycaret_seconds = time.perf_counter() - pycaret_start
print(f"PyCaret workflow time: {pycaret_seconds:.1f} s\n")


# =============================================================================
# Part B: scikit-learn workflow (rebuild the best PyCaret model by hand)
# =============================================================================
sklearn_start = time.perf_counter()

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# PyCaret model class name -> equivalent scikit-learn-compatible estimator.
# Anything not listed falls back to LightGBM (works inside a sklearn Pipeline).
MODEL_MAP = {
    "LGBMClassifier": lambda: LGBMClassifier(random_state=42, verbose=-1),
    "GradientBoostingClassifier": lambda: GradientBoostingClassifier(random_state=42),
    "RandomForestClassifier": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
    "ExtraTreesClassifier": lambda: ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "AdaBoostClassifier": lambda: AdaBoostClassifier(random_state=42),
    "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "GaussianNB": lambda: GaussianNB(),
    "LinearDiscriminantAnalysis": lambda: LinearDiscriminantAnalysis(),
    "QuadraticDiscriminantAnalysis": lambda: QuadraticDiscriminantAnalysis(),
    "RidgeClassifier": lambda: RidgeClassifier(random_state=42),
    "KNeighborsClassifier": lambda: KNeighborsClassifier(n_jobs=-1),
}
best_name = type(best).__name__
model_name = best_name if best_name in MODEL_MAP else "LGBMClassifier"
print(f"scikit-learn model: {model_name}")

X = df.drop(columns=[TARGET])
y = df[TARGET]  # already 1 = subscribes, 0 = not

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

model = Pipeline([("preprocess", preprocess), ("model", MODEL_MAP[model_name]())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("scikit-learn classification report (held-out 30% test set):")
print(classification_report(y_test, y_pred, target_names=["no", "yes"], digits=3))
print("Confusion matrix [rows=actual no/yes, cols=predicted no/yes]:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "model.joblib")
print("Saved model.joblib")

sklearn_seconds = time.perf_counter() - sklearn_start
print(f"scikit-learn workflow time: {sklearn_seconds:.1f} s")

print("\n=== Summary ===")
print(f"PyCaret:      {pycaret_seconds:6.1f} s")
print(f"scikit-learn: {sklearn_seconds:6.1f} s")
