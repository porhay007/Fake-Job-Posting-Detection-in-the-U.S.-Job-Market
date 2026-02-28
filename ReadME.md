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

Initial exploration of the dataset reveals several interesting patterns:

- Fraudulent job postings often contain missing or vague information.
- Legitimate postings are more likely to include company logos and screening questions.
- Some job attributes appear more frequently in fraudulent postings.

These insights will guide the feature engineering and modeling stages of the project.

---

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

The next phase of the project will include:

- Text preprocessing and feature extraction
- Feature engineering
- Building machine learning models
- Evaluating models using classification metrics
- Preparing the final project report and presentation

---

# References

Fake Job Postings Dataset  
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction