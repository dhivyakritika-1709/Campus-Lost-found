"""
matcher.py
Core Matching Engine for Campus Lost & Found System.
Loads the trained Logistic Regression model and computes genuine AI/ML match probabilities.
Supports bidirectional matching:
- LOST report matched against all FOUND reports.
- FOUND report matched against all LOST reports.
Sorts candidate matches by descending probability and generates grounded feature explanations.
"""

import os
import joblib
import numpy as np

from database import get_report_by_id, get_reports_for_matching
from features import extract_features_pair, explain_match, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.pkl")

_CACHED_MODEL_BUNDLE = None


def load_model():
    """
    Loads and caches the trained model bundle from data/model.pkl.
    Returns (model, metadata_dict).
    """
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE

    if not os.path.exists(MODEL_PATH):
        # Auto-train if dataset exists or notify
        try:
            from train_model import train
            print("Model not found. Auto-training now...")
            train()
        except Exception as e:
            print("Failed to auto-train model:", e)
            return None, None

    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            meta = bundle
        else:
            model = bundle
            meta = {"metrics": {}}

        _CACHED_MODEL_BUNDLE = (model, meta)
        return _CACHED_MODEL_BUNDLE
    except Exception as e:
        print("Error loading model from disk:", e)
        return None, None


def predict_pair_match_probability(report_a, report_b):
    """
    Calculates ML match probability between two specific reports.
    Returns (probability: float 0.0-1.0, feat_dict: dict, explanations: list[str]).
    """
    model, _ = load_model()
    feat_dict, feat_vec = extract_features_pair(report_a, report_b)

    if model is None:
        # Fallback weighted heuristic
        ml_prob = (
            feat_dict["name_similarity"] * 0.35 +
            feat_dict["category_similarity"] * 0.25 +
            feat_dict["colour_similarity"] * 0.20 +
            feat_dict["location_similarity"] * 0.10 +
            feat_dict["date_proximity"] * 0.10
        )
    else:
        # Existing Logistic Regression model uses the original 7 features
        prob_matrix = model.predict_proba([feat_vec])
        ml_prob = float(prob_matrix[0][1])

    # ---------------------------------------------------------
    # Visual similarity enhancement
    # ---------------------------------------------------------
    image_similarity = feat_dict.get("image_similarity", 0.5)

    # If either image is unavailable, image_similarity is neutral (0.5).
    # Only give visual similarity a modest influence so the original
    # ML model remains the primary matching engine.
    if image_similarity != 0.5:
        prob = (ml_prob * 0.80) + (image_similarity * 0.20)
    else:
        prob = ml_prob

    explanations = explain_match(feat_dict)
    return prob, feat_dict, explanations


def find_matches_for_report(target_report, candidate_reports, threshold=0.30):
    """
    Compares a single target report against a list of candidate reports (of opposite type).
    Filters and returns potential matches sorted in descending probability order.
    Never claims certainty; returns 'Potential Matches' with explicit probabilities.
    """
    matches = []

    for candidate in candidate_reports:
        # Do not compare a report to itself
        if candidate.get("id") == target_report.get("id"):
            continue

        prob, feat_dict, explanations = predict_pair_match_probability(target_report, candidate)

        if prob >= threshold:
            percentage = int(round(prob * 100))
            if percentage >= 80:
                confidence_tier = "High Potential Match"
            elif percentage >= 55:
                confidence_tier = "Moderate Potential Match"
            else:
                confidence_tier = "Possible Match"

            matches.append({
                "candidate": candidate,
                "probability": prob,
                "percentage": percentage,
                "confidence_tier": confidence_tier,
                "features": feat_dict,
                "explanations": explanations
            })

    # Sort in descending order of ML probability
    matches.sort(key=lambda m: m["probability"], reverse=True)
    return matches


def find_matches_by_id(report_id, threshold=0.30):
    """
    Convenience method: fetches a report from the database and retrieves its
    opposite-type candidates, running the ML matcher on all of them.
    """
    target = get_report_by_id(report_id, include_private=False)
    if not target:
        return []

    target_type = target.get("report_type", "LOST").upper()
    opposite_type = "FOUND" if target_type == "LOST" else "LOST"

    candidates = get_reports_for_matching(opposite_type)
    return find_matches_for_report(target, candidates, threshold=threshold)


if __name__ == "__main__":
    model, meta = load_model()
    print("Model loaded:", model)
    if meta:
        print("Metrics:", meta.get("metrics"))

    # Test with sample ID #1 (Lost bottle) against Found items
    matches = find_matches_by_id(1)
    print(f"\nFound {len(matches)} potential matches for Report #1:")
    for m in matches:
        print(f" -> Found Item #{m['candidate']['id']} ({m['candidate']['item_name']}): {m['percentage']}% ({m['confidence_tier']})")
        print("    Reasons:", m["explanations"])
