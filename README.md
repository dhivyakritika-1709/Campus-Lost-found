# Campus Lost & Found Management System

An AI/ML-based Campus Lost & Found Management System developed using Python, Scikit-Learn, Streamlit, and SQLite. The system allows users to report lost and found items and automatically identifies potential matches using machine learning.

## Project Overview

The system addresses common problems in traditional campus lost-and-found systems:

* Manual searching of lost and found reports
* Difficulty matching reports with different descriptions
* Public exposure of contact information
* Lack of a centralized lost-and-found platform

The system provides:

* Lost and found item reporting
* AI/ML-based matching
* Bidirectional LOST ↔ FOUND matching
* Contact request and privacy management
* SQLite-based data storage
* Image upload and storage
* Successful connection tracking

## Technologies Used

* Python
* Streamlit
* Scikit-Learn
* SQLite
* Pandas
* NumPy
* Joblib

## Machine Learning

The system uses **Logistic Regression** to calculate the probability that a lost and found report represents the same item.

Seven features are extracted from each pair of reports:

| Feature               | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `name_similarity`     | TF-IDF character n-gram similarity between item names   |
| `desc_similarity`     | TF-IDF word-level similarity between descriptions       |
| `keyword_overlap`     | Jaccard similarity between extracted keywords           |
| `category_similarity` | Similarity between item categories                      |
| `colour_similarity`   | Comparison of detected item colours                     |
| `location_similarity` | Similarity between reported locations                   |
| `date_proximity`      | Similarity based on the difference between report dates |

The extracted feature vector is passed to the trained Logistic Regression model using `predict_proba()`.

### Matching Pipeline

```text
Lost/Found Reports
        ↓
Feature Extraction
        ↓
7 Numerical Features
        ↓
Logistic Regression
        ↓
Match Probability
        ↓
Ranked Matches
```

Both training and prediction use the same feature extraction function to maintain consistency between the training and inference pipelines.

## Dataset and Model

The project uses a synthetically generated dataset for model training.

* 1,600 training samples
* Balanced matching and non-matching pairs
* 16 item categories
* 80/20 train-test split
* Logistic Regression classifier

### Model Performance

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  99.69% |
| Precision |  99.38% |
| Recall    | 100.00% |
| F1 Score  |  99.69% |

These results are based on synthetic data and may differ when the system is evaluated using real-world campus data.

## Privacy

Contact information is not publicly displayed.

The application uses a contact request system:

* Contact requests initially have `PENDING` status.
* The recipient can accept or decline the request.
* Contact details are shared only after acceptance.
* Identifying details are kept private.
* Public report queries exclude private contact information.

Database operations use parameterized SQL queries to reduce the risk of SQL injection.

## Project Structure

```text
Campus-Lost-and-Found/
│
├── app.py
├── database.py
├── features.py
├── matcher.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── uploads/
│
└── data/
    ├── lost_found.db
    ├── training_data.csv
    └── model.pkl
```

### File Description

| File                     | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| `app.py`                 | Streamlit application and user interface           |
| `database.py`            | SQLite database operations                         |
| `features.py`            | Feature extraction and similarity calculations     |
| `matcher.py`             | ML-based matching and prediction                   |
| `generate_dataset.py`    | Generates synthetic training data                  |
| `train_model.py`         | Trains and evaluates the Logistic Regression model |
| `requirements.txt`       | Python dependencies                                |
| `data/lost_found.db`     | Application database                               |
| `data/training_data.csv` | Generated training dataset                         |
| `data/model.pkl`         | Trained ML model                                   |
| `uploads/`               | Uploaded item images                               |

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Campus-Lost-and-Found
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Training Dataset

```bash
python generate_dataset.py
```

This creates:

```text
data/training_data.csv
```

### 4. Train the Model

```bash
python train_model.py
```

This creates:

```text
data/model.pkl
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Important

Run the following commands **before starting the Streamlit application**:

```bash
python generate_dataset.py
python train_model.py
streamlit run app.py
```

## Limitations

* The current model is trained using synthetic data.
* Uploaded images are stored and displayed but are not directly used for ML-based image similarity.
* Real-world deployment would require authentication, moderation, notifications, and additional security measures.
* Model performance on real campus data may differ from the reported synthetic-data results.

## Future Improvements

* Image-based similarity using computer vision
* Real campus dataset for model training
* Student authentication
* Email and notification system
* Admin verification
* Improved NLP/embedding-based matching
* Online deployment

## Purpose

This project was developed as an academic project to explore the application of machine learning, web development, database management, and privacy-aware system design to a practical campus problem.
