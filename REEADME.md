# Fake Job Posting Detection in the U.S. Job Market

## Team Members

- Porhay Rouen
- MuyKhim Ing

---

## Project Overview

Fraudulent job postings have become increasingly common in online U.S. job platforms, targeting job seekers with fake promises, identity theft, and financial scams. This project builds an end-to-end machine learning pipeline to automatically classify job postings as **legitimate or fraudulent**.

The pipeline combines:
- **Structured features** (employment type, industry, company logo, etc.) via one-hot encoding
- **Text features** (job description + requirements) via TF-IDF vectorization
- **XGBoost classifier** selected after comparing 3 models
---

## Problem Statement

Fraudulent job postings waste applicants' time, cause financial harm, and erode trust in online job platforms. This project aims to:

- Examine characteristics that distinguish fraudulent postings from legitimate ones
- Build a predictive ML model using both structured and textual features
---

## Dataset

**Source:** Kaggle — Real / Fake Job Posting Prediction  
**URL:** https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

| Detail | Value |
|--------|-------|
| Original size | 17,880 postings |
| After U.S. filter | 10,656 postings |
| Legitimate | 9,926 (93.1%) |
| Fraudulent | 730 (6.9%) |
| Storage | SQLite database (`job_postings.db`) |

**Key features used:**

| Feature | Type |
|---------|------|
| `employment_type` | Categorical |
| `industry` | Categorical |
| `department` | Categorical |
| `required_education` | Categorical |
| `required_experience` | Categorical |
| `has_company_logo` | Binary |
| `telecommuting` | Binary |
| `description` | Text (TF-IDF) |
| `requirements` | Text (TF-IDF) |
| `fraudulent` | Target (0/1) |

---

## Technologies Used

- **Python** — pandas, numpy, scikit-learn, xgboost, joblib, scipy
- **NLP** — TF-IDF Vectorization (500 features, bigrams)
- **Database** — SQLite (relational DB bonus component)
- **Web** — Flask API, HTML/CSS/JavaScript
- **Visualization** — Matplotlib, Seaborn
- **Notebook** — Jupyter Notebook

---

## Project Structure

```
Group_Project/
│
├── Data/
│   └── fake_job_postings.csv
│
├── Coding/
│   ├── Index.ipynb              ← Main notebook (full pipeline)
│   └── job_postings.db          ← SQLite database
│
├── Requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Filtered to **U.S. postings only** (`location` contains `'US'`)
- Filled missing categorical values with `'Unknown'` / `'Not Specified'`
- Removed 203 duplicate records
- Stored in **SQLite** for structured SQL querying

### 2. Feature Engineering
- **Structured features** → `pd.get_dummies()` → 878 columns
- **Text features** → TF-IDF on `description + requirements` → 500 columns
- **Combined** → `scipy.sparse.hstack` → 1,378 total features

### 3. Modeling
Three models were trained and compared:

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|-------|----------|----------------|--------------|----------|
| Random Forest | 97% | 100% | 53% | 0.69 |
| Logistic Regression | 94% | 53% | 91% | 0.67 |
| **XGBoost** ⭐ | **97%** | **87%** | **75%** | **0.80** |

**XGBoost was selected** as the final model — best balance of precision and recall, highest Fraud F1-score.

### 4. Model Configuration
```python
XGBClassifier(
    n_estimators=100,
    scale_pos_weight=13,   # handles 93/7 class imbalance
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
# Decision threshold tuned to 0.35 for better fraud recall
# 5-fold stratified cross-validation used for evaluation
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost flask flask-cors joblib scipy matplotlib seaborn
brew install libomp   # macOS only — required for XGBoost
```

### Step 2 — Run the notebook
Open and run `Index.ipynb` top to bottom. This generates:
- `xgb_model.pkl`
- `tfidf_vectorizer.pkl`
- `model_features.pkl`

### Step 3 — Start the Flask server
```bash
cd Coding/
python3 app.py
```

### Step 4 — Open the web interface
Open your browser and go to:
```
http://127.0.0.1:5000
```

Paste any job posting description and click **Analyze Job Posting** to get an instant fraud prediction.

---

## Web Interface Features

- 🤖 **XGBoost + TF-IDF** real-time predictions via Flask REST API
- 📊 **Fraud probability score** with visual confidence bar
- 🏢 **Structural signals** — logo, education, experience, industry checks
- ⚠️ **14 red flag keyword patterns** scanned from job text
- 🔗 **LinkedIn-style UI** — professional white & blue design

---

## Key Findings

- **Company logo presence** is the strongest single structured predictor — fraudulent postings rarely include a logo
- **TF-IDF text features** significantly improved fraud recall over structured features alone
- **XGBoost** outperformed Random Forest (F1: 0.80 vs 0.69) and Logistic Regression (F1: 0.80 vs 0.67)
- **Threshold tuning to 0.35** improved fraud detection recall beyond the default 0.5

---

## Challenges

| Challenge | Solution |
|-----------|----------|
| Class imbalance (93% legit / 7% fraud) | `scale_pos_weight=13` + threshold tuning |
| Low recall on fraud (Random Forest = 53%) | Switched to XGBoost + added TF-IDF text features |
| `scipy` dtype error with object columns | Applied `.astype(float)` before `csr_matrix` |
| CORS error when opening HTML directly | Served interface through Flask via `send_from_directory` |

---

## References

- Bansal, S. (2020). *Real / Fake Job Posting Prediction*. Kaggle. https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
