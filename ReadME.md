# Fake Job Posting Detection in the U.S. Job Market

## Team Members

- Porhay Rouen
- MuyKhim Ing

---

# Project Overview

Fraudulent job postings have become increasingly common in online job platforms. These scams often target job seekers with misleading opportunities, fake company information, or attempts to collect personal and financial data.

This project analyzes job posting data to understand the characteristics of fraudulent job listings and explores how different job attributes relate to the likelihood of fraud. Through data preprocessing and exploratory data analysis (EDA), the project identifies patterns and insights that may help detect fraudulent job postings.

---

# Problem Statement

Fraudulent job postings are increasingly common in the U.S., targeting job seekers with fake promises, scams, or identity theft. These postings waste applicants’ time, can cause financial loss, and harm the credibility of online job platforms.

This project aims to:

- Examine characteristics of fraudulent postings
- Explore relationships between structured and textual job features
- Prepare features for building predictive models capable of classifying job postings as real or fraudulent

---

# Dataset

The dataset used in this project is publicly available on Kaggle.

Fake Job Postings Dataset:  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

The dataset contains job postings collected from online platforms and includes both legitimate and fraudulent listings. It contains both structured attributes and textual descriptions of job postings.

Example features include:

- Job title
- Location
- Employment type
- Industry
- Department
- Salary range
- Job description
- Job requirements
- Fraud label (0 = Real, 1 = Fraudulent)

---

# Technologies Used

This project uses the following tools and technologies:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SQLite
- Jupyter Notebook

SQLite is used as a relational database to store and query the dataset, which supports the SQL component of the project.

---

# Data Preprocessing

Several preprocessing steps were performed before conducting analysis:

### Data Cleaning

- Handling missing values
- Removing duplicate records
- Replacing missing categorical values with "Unknown"

### Feature Selection

Selected relevant columns for analysis including:

- title
- location
- employment_type
- industry
- department
- salary_range
- description
- requirements
- fraudulent

### Data Storage

The dataset was stored in a SQLite database to allow structured SQL queries and relational data management.

---

# Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand patterns in the dataset.

### Distribution Analysis

The distribution of fraudulent vs legitimate job postings was analyzed. The dataset is highly imbalanced, with far fewer fraudulent postings compared to legitimate ones.

### Feature Relationships

Relationships between several job attributes and fraud labels were explored, including:

- Telecommuting vs Fraud
- Company Logo vs Fraud
- Required Education vs Fraud

### Correlation Analysis

A correlation matrix was generated to examine relationships among numeric features and their potential association with fraudulent postings.

---

# Preliminary Findings

From the exploratory analysis, several patterns emerge:

- Fraudulent job postings often contain less detailed information

- Some industries appear more frequently in fraudulent listings

- Fraud detection datasets are highly imbalanced

- Missing or vague job descriptions may indicate suspicious postings

## These insights will guide feature engineering and model development in the next phase.

# Project Structure

```
Group Project/
│
├── Data/
│ └── fake_job_postings.csv
│
├── Coding/
│ ├── Index.ipynb
│ └── job_postings.db
│
└── README.md
```

---

# Next Steps

The remaining tasks for the project include:

## Remaining Data Processing

- Additional feature engineering

- Text feature extraction from job descriptions and requirements

## Modeling

- Build classification models such as:

  - Logistic Regression

  - Random Forest

  - Decision Trees

## Evaluation

Models will be evaluated using:

- Accuracy

- Precision

- Recall

- F1-score

## Documentation

- Prepare final report

- Create presentation slides

- Document code and results

---

# References

Fake Job Postings Dataset  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
