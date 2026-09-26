# Campus Lost & Found Management System

An AI/ML-based Campus Lost & Found Management System designed for 2nd-year B.Tech Computer Science & Engineering students. Built using Python, Scikit-Learn, Streamlit, and SQLite.

---

## 📌 Project Objective
On a college campus, students frequently misplace valuable belongings (smartphones, ID cards, water bottles, keys, calculators, laptop chargers, etc.). Traditional physical lost-and-found notice boards or WhatsApp group messages suffer from:
1. **No automated matching**: Students must manually read hundreds of messages.
2. **Privacy risks**: Phone numbers posted publicly on campus notice boards or groups lead to spam, pranks, and harassment.
3. **Phording / Inconsistent phrasing**: A lost report might say *"Blue Milton thermos"* while the finder describes it as *"Blue steel water bottle"*. Simple exact keyword search fails.

This project solves these issues with:
- **Bidirectional AI/ML Matching**: Automatically pairs **LOST ↔ FOUND** reports using machine learning.
- **Genuine Logistic Regression**: Computes real continuous match probabilities via `predict_proba()`.
- **Feature Engineering**: Combines TF-IDF n-gram cosine similarities, keyword Jaccard overlap, category clustering, color distance, and exponential date decay.
- **Contact Request Privacy System**: Student phone numbers are **never displayed publicly**. Contact details are only exchanged when a finder/owner mutually accepts an incoming contact request.

---

## 📁 Strict Project File Structure

```text
Campus-Lost-and-Found/
│
├── app.py                  # Full Streamlit Web UI and User Interaction System
├── database.py             # SQLite persistence, parameterized queries & privacy logic
├── features.py             # Feature engineering & TF-IDF similarity extraction
├── matcher.py              # ML inference pipeline using Logistic Regression predict_proba()
├── generate_dataset.py     # Synthetic balanced training dataset generator (1600 pairs)
├── train_model.py          # Supervised training & evaluation (Accuracy, F1, Coefficients)
├── README.md               # Complete architectural documentation & viva guide
├── requirements.txt        # Python dependency declarations
│
├── uploads/                # Safe storage for uploaded item images
│
└── data/
    ├── lost_found.db       # SQLite database (reports & contact_requests tables)
    ├── training_data.csv   # Synthetic demo training pairs (16 item categories)
    └── model.pkl           # Trained Logistic Regression model bundle (joblib)
```

---

## 🧠 Machine Learning Architecture

### 1. Why Logistic Regression?
Unlike arbitrary `if-else` heuristic score calculators, this system uses `sklearn.linear_model.LogisticRegression`.
- The model learns an optimal hyperplane in 7-dimensional feature space separating matching pairs from non-matching pairs.
- The standard logistic (sigmoid) function maps the linear combination of weighted features to a calibrated probability between 0.0 and 1.0:

$$P(\text{Match} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where:
- $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]^T$ is the feature vector.
- $\mathbf{w}$ is the learned weight vector.
- $b$ is the model intercept (bias).

### 2. Feature Engineering Pipeline (`features.py`)
For every pair of reports, seven numerical comparison features are calculated:

| # | Feature Name | Description | Mathematical / Algorithmic Basis |
|---|---|---|---|
| $x_1$ | `name_similarity` | Item Name Similarity | Character n-gram (2–4) TF-IDF Cosine Similarity. Resilient to typos and abbreviations. |
| $x_2$ | `desc_similarity` | Description Similarity | Word unigram/bigram TF-IDF Cosine Similarity over item description text. |
| $x_3$ | `keyword_overlap` | Keyword Jaccard Overlap | Token set intersection divided by token set union: $\frac{\|A \cap B\|}{\|A \cup B\|}$ |
| $x_4$ | `category_similarity` | Category Cluster Match | 1.0 for exact category match; 0.75 for semantic clusters; 0.0 otherwise. |
| $x_5$ | `colour_similarity` | Colour Similarity | Token dictionary color extraction. Matches = 1.0, clashing colors = 0.0. |
| $x_6$ | `location_similarity` | Campus Location Overlap | Token overlap across campus buildings, rooms, and landmarks. |
| $x_7$ | `date_proximity` | Date Time Proximity | Exponential decay function: $e^{-\frac{\|\text{day}_1 - \text{day}_2\|}{7.0}}$ (Same day = 1.0). |

Both model training (`train_model.py`) and live inference (`matcher.py`) invoke the identical `extract_features_pair()` function, preventing train-serve skew.

### 3. Learned Feature Weights
During training on 1,600 balanced samples:
- **Model Intercept (Bias)**: `-9.6743`
- **Name Similarity Weight**: `+4.9096`
- **Colour Similarity Weight**: `+3.3856`
- **Category Similarity Weight**: `+3.3568`
- **Date Proximity Weight**: `+3.3427`
- **Keyword Overlap Weight**: `+2.9646`
- **Location Similarity Weight**: `+1.9633`
- **Description Similarity Weight**: `+0.8719`

Test Set Performance:
- **Accuracy**: 99.69%
- **Precision**: 99.38%
- **Recall**: 100.00%
- **F1-Score**: 99.69%

---

## 🔄 Bidirectional Matching (LOST ↔ FOUND)

The matching pipeline works symmetrically:

1. **LOST ➔ FOUND Matching**:
   When a student reports losing an item (e.g. *Black Milton Bottle*):
   - The system retrieves all active `FOUND` items from SQLite.
   - Computes features for each candidate against the lost report.
   - Evaluates `model.predict_proba()`.
   - Ranks matches in descending order and displays grounded reasons (e.g., *"✓ Same primary colour", "✓ Close occurrence dates"*).

2. **FOUND ➔ LOST Matching**:
   When a student or security guard reports finding an item:
   - The system retrieves all active `LOST` reports from SQLite.
   - Computes features for each candidate against the found item.
   - Alerts the finder to registered lost item owners who may be searching for it.

---

## 🔒 Contact Request & Student Privacy System

### Why Privacy by Default?
Naively displaying phone numbers publicly exposes students to:
- Spam marketing calls.
- Fake claimers trying to claim valuable electronics.
- Stalking and social engineering.

### Privacy Enforcement Architecture
1. **Public Views**:
   - `get_all_reports()` strictly excludes `contact_number` and `identifying_details`.
   - Public cards display: *"Contact details are private and shared only after a contact request is accepted."*
2. **Contact Requests State Machine**:
   - Status starts as `PENDING`.
   - Target recipient reviews the incoming request in **My Reports & Requests**.
   - If **ACCEPTED**: Both parties' phone numbers are revealed for direct calling/messaging.
   - If **DECLINED**: The phone number remains permanently protected.
3. **Secret Verification Details**:
   - Reporters can enter an identifying secret (e.g., *"Dent on base, stickers inside back cover"*).
   - This detail is kept confidential until personal handover to prove genuine ownership.

---

## 🛠️ Step-by-Step Installation & Setup Commands

### 1. Clone / Open the Project
```bash
cd Campus-Lost-and-Found
```

### 2. Install Required Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the Synthetic Training Dataset
Generates 1,600 balanced pairs across 16 campus item categories:
```bash
python generate_dataset.py
```
*(Outputs `data/training_data.csv`)*

### 4. Train the Machine Learning Model
Fits the Logistic Regression model and computes evaluation metrics:
```bash
python train_model.py
```
*(Outputs `data/model.pkl`)*

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` (or port `3000` depending on container configuration).

---

## 🎓 Viva Voce & CSE Interview Questions & Answers

### Q1: Why did you choose Logistic Regression over a Deep Neural Network?
**Answer**: For tabular comparison vectors derived from feature engineering, Logistic Regression is computationally efficient, explainable, and does not require GPU acceleration or large deep learning dependencies. Its coefficients directly reflect feature importance, making the system transparent and easily auditable.

### Q2: How does the model output a probability rather than just a 0 or 1?
**Answer**: Logistic Regression computes the log-odds (logit): $z = \mathbf{w}^T \mathbf{x} + b$. By applying the standard sigmoid function $\frac{1}{1 + e^{-z}}$, $z \in (-\infty, \infty)$ is mapped to $[0, 1]$, representing $P(y=1 \mid \mathbf{x})$. Scikit-Learn exposes this via `predict_proba()`.

### Q3: What is the purpose of character n-gram TF-IDF?
**Answer**: Word-level matching fails when users make spelling typos or use compound words (e.g., *"EarPods"* vs *"ear pods"*, *"GalaxyS21"* vs *"Galaxy S21"*). Character n-grams (sub-tokens of length 2 to 4) capture substring overlap regardless of spacing or minor typos.

### Q4: How is data leakage prevented between train and test splits?
**Answer**: The synthetic dataset is generated into distinct independent report pairs. The train-test split (80/20) uses a fixed random seed and stratification on the target label $y$ before model fitting. No test labels or test vectors influence training.

### Q5: How is SQL injection prevented?
**Answer**: In `database.py`, all SQL queries use parameterized syntax (`?` placeholders). User inputs are sent separately to the SQLite database engine as parameter tuples, preventing malicious SQL code injection.

---

## ⚠️ Notes & Limitations
- **Synthetic Training Data**: The dataset in `data/training_data.csv` consists of generated synthetic pairs representing typical campus scenarios. Real-world institutional deployment would incorporate anonymized historical campus data.
- **Image Similarity**: Images are safely stored in `uploads/` and rendered in report cards. For high performance and lightweight execution on standard student laptops, matching is executed on text and attribute features rather than heavy convolutional networks.
