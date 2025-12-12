# Student Grades Prediction

**Course:** Machine Learning (MMN1) - A4 ESILV
**Date:** December 2025

## Project Overview

In the modern educational landscape, understanding the factors that drive academic success is crucial. This project aims to build a predictive machine learning model to estimate a student's final exam score based on various demographic, social, and academic factors.

We treated this as a regression problem, utilizing a dataset of ~6,600 students to identify influential factors and predict the continuous target variable: `Exam_Score`.

## Authors

* **Daphne BARAY**
* **Sofiane BEAUMONT**
* **Max BAUMBERGER**

## Useful Links

* **Kaggle Dataset:** [Student Performance Factors](https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights/data)
* **GitHub Repository:** [View Source Code](https://github.com/maxbaumberger/gradesprediction/)
* **Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1d1VkLdJjlsVYWFKxfuNtNhik5k-nF-bh)

## The Dataset

**Source:** [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights/data)
**Size:** ~6,600 observations, 20 features.

### Key Features
The dataset includes both numerical and categorical variables:

* **Academic Habits:** `Hours_Studied`, `Attendance`, `Tutoring_Sessions`, `Previous_Scores`.
* **Demographics & Social:** `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Family_Income`, `Teacher_Quality`, etc.
* **Target:** `Exam_Score` (0-100).

## Methodology & Pipeline

### 1. Data Cleaning & Preprocessing
* **Missing Values:** Handling missing data in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` using Mode Imputation.
* **Feature Selection:** Dropped variables with a correlation coefficient < 0.1 with the target to reduce noise.
* **Encoding:** Applied One-Hot Encoding to categorical variables.
* **Scaling:** Applied StandardScaler to normalize numerical features.
* **Split:** 70% Training / 30% Testing (random_state=42).

### 2. Models Implemented
We implemented a progressive modeling strategy:
1.  **Decision Tree Regressor (Baseline):** Optimized using `GridSearchCV`.
2.  **Bagging Regressor:** Ensemble method using 100 decision trees to reduce variance.
3.  **XGBoost (Extreme Gradient Boosting):** Advanced model optimized using `RandomizedSearchCV`.

## Results

We evaluated the models based on R² Score, Mean Absolute Error (MAE), and RMSE.

| Model | R² Score | MAE | RMSE | Performance |
| :--- | :---: | :---: | :---: | :--- |
| **Decision Tree (Baseline)** | 0.59 | ~ | ~ | Baseline performance, struggled with complexity. |
| **Bagging Regressor** | 0.67 | ~ | ~ | Improved robustness by averaging predictions. |
| **XGBoost (Optimized)** | **0.70** | **1.04** | **2.03** | **Best Model.** High precision and low error. |

**Key Insight:** Academic habits like **Attendance** and **Hours Studied** were the strongest predictors. The final XGBoost model can predict a student's score with an average error margin of only ~1 point.

## How to Run

### Prerequisites
Make sure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn