"""
features.py
Feature Engineering for Campus Lost & Found ML System.
Extracts 7 numerical similarity and proximity features between two reports:
1. Item Name Similarity (Character & Word TF-IDF Cosine Similarity)
2. Description Similarity (TF-IDF Cosine Similarity)
3. Keyword Overlap (Token Jaccard Intersection)
4. Category Similarity (Exact & Semantic category mapping)
5. Colour Similarity (Color token detection and matching)
6. Location Similarity (Location token overlap)
7. Date Proximity (Exponential time decay based on calendar days)

Guarantees identical feature vector generation for both training and inference.
"""

import math
import re
import os
from datetime import datetime
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_NAMES = [
    "name_similarity",
    "desc_similarity",
    "keyword_overlap",
    "category_similarity",
    "colour_similarity",
    "location_similarity",
    "date_proximity"
]

COLOR_SET = {
    "black", "white", "blue", "navy", "red", "green", "yellow", "orange",
    "purple", "pink", "grey", "gray", "silver", "gold", "brown", "beige",
    "transparent", "maroon", "teal", "cyan", "dark", "light"
}

CATEGORY_MAP = {
    "Electronics": {"electronics", "gadgets", "phone", "laptop", "charger", "earphones", "headphones"},
    "Daily Essentials": {"essentials", "bottle", "umbrella", "lunchbox", "flask", "flask"},
    "Personal Belongings": {"belongings", "wallet", "keys", "watch", "spectacles", "glasses", "ring"},
    "Campus Specific": {"campus", "id card", "identity", "library", "lab", "rfid"},
    "Study Materials": {"study", "books", "notes", "notebook", "calculator", "binder", "bag", "stationery"},
    "Bags & Luggage": {"bag", "backpack", "pouch", "luggage", "tote"}
}


def _clean_text(text):
    """Lowercases, removes punctuation, and standardizes whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _calc_text_cosine(text1, text2, analyzer="char_wb", ngram_range=(2, 4)):
    """Computes TF-IDF Cosine Similarity between two strings."""
    c1 = _clean_text(text1)
    c2 = _clean_text(text2)
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0

    try:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        matrix = vec.fit_transform([c1, c2])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Fallback to simple token intersection
        s1, s2 = set(c1.split()), set(c2.split())
        if not s1 or not s2:
            return 0.0
        return float(len(s1 & s2) / len(s1 | s2))


def _calc_keyword_overlap(text1, text2):
    """Calculates Jaccard overlap of meaningful alphanumeric tokens."""
    c1 = set(_clean_text(text1).split())
    c2 = set(_clean_text(text2).split())
    stop_words = {"the", "a", "an", "in", "on", "at", "for", "with", "and", "or", "of", "to", "is", "it", "my", "was"}
    c1 = {w for w in c1 if w not in stop_words and len(w) > 1}
    c2 = {w for w in c2 if w not in stop_words and len(w) > 1}

    if not c1 or not c2:
        return 0.0
    intersection = c1 & c2
    union = c1 | c2
    return float(len(intersection) / len(union))


def _calc_category_similarity(cat1, cat2):
    """Matches categories with tolerance for semantic synonyms."""
    c1 = _clean_text(cat1)
    c2 = _clean_text(cat2)
    if not c1 or not c2:
        return 0.2
    if c1 == c2:
        return 1.0

    # Check semantic cluster
    for _, keywords in CATEGORY_MAP.items():
        in_1 = any(kw in c1 for kw in keywords)
        in_2 = any(kw in c2 for kw in keywords)
        if in_1 and in_2:
            return 0.75

    return 0.0


def _extract_colors(text):
    """Extracts known color tokens from text."""
    words = set(_clean_text(text).split())
    return words & COLOR_SET


def _calc_colour_similarity(col1, col2, desc1="", desc2=""):
    """Compares specified colors and mentions in descriptions."""
    set1 = _extract_colors(col1) | _extract_colors(desc1)
    set2 = _extract_colors(col2) | _extract_colors(desc2)

    # If neither mentioned a color, neutral match
    if not set1 and not set2:
        return 0.5
    # If one mentioned color and one didn't, slight penalty
    if not set1 or not set2:
        return 0.4
    # If they share at least one color
    if set1 & set2:
        return 1.0
    # Clashing colors (e.g. blue vs red)
    return 0.0


def _calc_location_similarity(loc1, loc2):
    """Compares campus location strings."""
    c1 = _clean_text(loc1)
    c2 = _clean_text(loc2)
    if not c1 or not c2:
        return 0.3
    if c1 == c2:
        return 1.0

    tokens1 = set(c1.split())
    tokens2 = set(c2.split())
    stop_tokens = {"block", "floor", "room", "hall", "near", "at", "the", "in", "dept", "department"}
    t1 = {t for t in tokens1 if t not in stop_tokens}
    t2 = {t for t in tokens2 if t not in stop_tokens}

    if not t1 or not t2:
        return 0.3

    overlap = t1 & t2
    if overlap:
        return float(len(overlap) / max(len(t1), len(t2)))
    return 0.0


def _calc_date_proximity(date_str1, date_str2):
    """
    Computes time proximity score using exponential decay:
    exp(-|days_diff| / 7.0).
    Same day = 1.0, 7 days = ~0.37, 30 days = < 0.02.
    """
    if not date_str1 or not date_str2:
        return 0.5

    try:
        # Support YYYY-MM-DD or common date formats
        d1 = datetime.strptime(str(date_str1).strip()[:10], "%Y-%m-%d")
        d2 = datetime.strptime(str(date_str2).strip()[:10], "%Y-%m-%d")
        days_diff = abs((d1 - d2).days)
        # 7-day half-life decay
        score = math.exp(-days_diff / 7.0)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.5

def _calc_image_similarity(image_path1, image_path2):
    """
    Computes visual similarity between two uploaded item images
    using perceptual image hashing.

    Returns a score between 0.0 and 1.0.
    Higher score = more visually similar.
    """

    if not image_path1 or not image_path2:
        return 0.5

    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        return 0.5

    try:
        img1 = Image.open(image_path1).convert("RGB").resize((32, 32))
        img2 = Image.open(image_path2).convert("RGB").resize((32, 32))

        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)

        # Normalize pixel values
        arr1 = arr1 / 255.0
        arr2 = arr2 / 255.0

        # Mean absolute pixel difference
        difference = np.mean(np.abs(arr1 - arr2))

        # Convert difference into similarity
        similarity = 1.0 - difference

        return float(np.clip(similarity, 0.0, 1.0))

    except Exception:
        return 0.5
    
def extract_features_pair(report_a, report_b):
    """
    Extracts the 7 numerical features between report_a and report_b.
    Works symmetrically (Lost vs Found or Found vs Lost).
    Returns a dictionary of named features and a list of floats.
    """
    name_a = report_a.get("item_name", "")
    name_b = report_b.get("item_name", "")

    desc_a = report_a.get("description", "")
    desc_b = report_b.get("description", "")

    cat_a = report_a.get("category", "")
    cat_b = report_b.get("category", "")

    col_a = report_a.get("colour", "")
    col_b = report_b.get("colour", "")

    loc_a = report_a.get("location", "")
    loc_b = report_b.get("location", "")

    date_a = report_a.get("date", "")
    date_b = report_b.get("date", "")

    f_name = _calc_text_cosine(name_a, name_b, analyzer="char_wb", ngram_range=(2, 4))
    f_desc = _calc_text_cosine(desc_a, desc_b, analyzer="word", ngram_range=(1, 2))
    f_keyword = _calc_keyword_overlap(f"{name_a} {desc_a}", f"{name_b} {desc_b}")
    f_cat = _calc_category_similarity(cat_a, cat_b)
    f_col = _calc_colour_similarity(col_a, col_b, desc_a, desc_b)
    f_loc = _calc_location_similarity(loc_a, loc_b)
    f_date = _calc_date_proximity(date_a, date_b)
    
    image_a = report_a.get("image_path", "")
    image_b = report_b.get("image_path", "")

    f_image = _calc_image_similarity(image_a, image_b)

    feature_dict = {
        "name_similarity": f_name,
        "desc_similarity": f_desc,
        "keyword_overlap": f_keyword,
        "category_similarity": f_cat,
        "colour_similarity": f_col,
        "location_similarity": f_loc,
        "date_proximity": f_date,
        "image_similarity": f_image
    }

    feature_vector = [feature_dict[name] for name in FEATURE_NAMES]
    return feature_dict, feature_vector


def explain_match(features_dict):
    """
    Generates human-readable, grounded explanations for why two reports match,
    based on the actual numerical feature values calculated.
    """
    explanations = []

    if features_dict.get("category_similarity", 0) >= 0.7:
        explanations.append("Matching item category")

    if features_dict.get("colour_similarity", 0) >= 0.8:
        explanations.append("Same primary colour identified")

    if features_dict.get("name_similarity", 0) >= 0.35:
        explanations.append("High similarity in item name / brand")

    if features_dict.get("desc_similarity", 0) >= 0.25:
        explanations.append("Consistent visual description details")

    if features_dict.get("keyword_overlap", 0) >= 0.25:
        explanations.append("Overlapping identifying keywords")

    if features_dict.get("location_similarity", 0) >= 0.3:
        explanations.append("Nearby campus loss & find location")

    if features_dict.get("date_proximity", 0) >= 0.5:
        explanations.append("Close occurrence dates (within days)")
    if features_dict.get("image_similarity", 0.5) >= 0.80:
        explanations.append("High visual similarity between uploaded item images")
    elif features_dict.get("image_similarity", 0.5) >= 0.65:
        explanations.append("Similar visual appearance in uploaded images")
    if not explanations:
        explanations.append("Potential match identified across composite features")

    return explanations


if __name__ == "__main__":
    # Test feature extraction on sample pair
    r1 = {
        "item_name": "Black Milton Water Bottle",
        "category": "Daily Essentials",
        "description": "Stainless steel flask with scratch on cap",
        "colour": "Black",
        "location": "Central Library 2nd Floor",
        "date": "2026-09-18"
    }
    r2 = {
        "item_name": "Milton Black Steel Bottle",
        "category": "Daily Essentials",
        "description": "Black flask found near reading table",
        "colour": "Black",
        "location": "Central Library Reading Hall",
        "date": "2026-09-19"
    }
    f_dict, vec = extract_features_pair(r1, r2)
    print("Feature Dict:", f_dict)
    print("Feature Vector:", vec)
    print("Explanations:", explain_match(f_dict))
