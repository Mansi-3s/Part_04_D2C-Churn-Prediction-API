import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# =====================================================
# Load Dataset
# =====================================================

DATA_PATH = "data/rfm_modeling_snapshot.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)
print("Shape:", df.shape)

# =====================================================
# Define Target
# =====================================================

TARGET = "churn_next_60d"

# Columns not used for training
DROP_COLUMNS = [
    "customer_id",
    "snapshot_date",
    "split",
    TARGET
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]

# =====================================================
# Feature Types
# =====================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(numeric_features)

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Preprocessing Pipelines
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# =====================================================
# Logistic Regression Pipeline
# =====================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# =====================================================
# Train Model
# =====================================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training Complete!")

# =====================================================
# Predictions
# =====================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, predictions))

# =====================================================
# Save Model
# =====================================================

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 50)
print("Model Saved Successfully")
print("=" * 50)
print(f"Location: {MODEL_PATH}")

import json
from sklearn.metrics import confusion_matrix
# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, predictions))

# =====================================================
# Save Metrics
# =====================================================

metrics = {
    "model": "Logistic Regression",
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "selected_threshold": 0.50,
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to metrics.json")