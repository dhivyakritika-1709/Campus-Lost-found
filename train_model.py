"""
train_model.py
Machine Learning Model Training Pipeline for Campus Lost & Found System.
1. Loads synthetic training pairs from data/training_data.csv.
2. Extracts numerical comparison features using features.py.
3. Splits into Train (80%) and Test (20%) sets.
4. Trains sklearn.linear_model.LogisticRegression.
5. Evaluates Accuracy, Precision, Recall, and F1-Score.
6. Saves trained model and metadata to data/model.pkl using joblib.
"""

import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from features import FEATURE_NAMES, extract_features_pair

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")


def load_and_extract_features():
    """Loads CSV and converts text pairs into a numerical feature matrix."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Training dataset not found at {CSV_PATH}. Run generate_dataset.py first.")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} training pairs from {CSV_PATH}")

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        lost_item = {
            "item_name": str(row.get("item_name_lost", "")),
            "category": str(row.get("category_lost", "")),
            "description": str(row.get("desc_lost", "")),
            "colour": str(row.get("colour_lost", "")),
            "location": str(row.get("location_lost", "")),
            "date": str(row.get("date_lost", ""))
        }
        found_item = {
            "item_name": str(row.get("item_name_found", "")),
            "category": str(row.get("category_found", "")),
            "description": str(row.get("desc_found", "")),
            "colour": str(row.get("colour_found", "")),
            "location": str(row.get("location_found", "")),
            "date": str(row.get("date_found", ""))
        }

        _, feat_vec = extract_features_pair(lost_item, found_item)
        X_list.append(feat_vec)
        y_list.append(int(row["is_match"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def train():
    """Trains Logistic Regression model and saves to disk."""
    print("=" * 60)
    print("CAMPUS LOST & FOUND - LOGISTIC REGRESSION TRAINING PIPELINE")
    print("=" * 60)

    X, y = load_and_extract_features()
    print(f"Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")

    # Train / Test split with stratification to preserve balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Initialize and train Logistic Regression
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- MODEL EVALUATION METRICS (TEST SET) ---")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc * 100:.2f}%")

    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["Non-Match (0)", "Match (1)"]))

    print("--- LEARNED LOGISTIC REGRESSION COEFFICIENTS ---")
    intercept = clf.intercept_[0]
    print(f"Model Intercept (Bias): {intercept:.4f}")
    for name, coef in zip(FEATURE_NAMES, clf.coef_[0]):
        print(f"  - {name:20s}: {coef:+.4f}")

    # Save model and metadata bundle
    os.makedirs(DATA_DIR, exist_ok=True)
    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "intercept": float(intercept),
        "coefficients": {name: float(c) for name, c in zip(FEATURE_NAMES, clf.coef_[0])},
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc)
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSUCCESS: Trained model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
