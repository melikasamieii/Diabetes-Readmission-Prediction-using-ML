# 🏥 Diabetes Readmission Prediction

A machine learning project for predicting whether a diabetic patient will be **readmitted to the hospital within 30 days** after discharge.

The project focuses on data preprocessing, feature engineering, handling class imbalance, machine learning model comparison, feature importance, and subgroup fairness analysis.

> **Note:** This project is intended for educational and research purposes only. It is not designed for clinical diagnosis or real-world medical decision-making.

---

## 📌 Project Overview

Hospital readmission is an important problem in healthcare because early readmissions can increase healthcare costs and indicate that additional follow-up or care may be required.

In this project, machine learning models are trained to predict whether a patient will be readmitted within **30 days**.

The original dataset contains three target categories:

* `<30` → Readmitted within 30 days
* `>30` → Readmitted after 30 days
* `NO` → Not readmitted

For this project, the target was converted into a binary classification problem:

| Target | Meaning                       |
| ------ | ----------------------------- |
| `1`    | Readmitted within 30 days     |
| `0`    | Not readmitted within 30 days |

---

## 📊 Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset.

Original dataset:

* **101,766 patient encounters**
* **50 features**
* Clinical, demographic, medication, diagnosis, and hospitalization-related information

The dataset includes information such as:

* Age
* Gender
* Race
* Time spent in hospital
* Number of diagnoses
* Number of medications
* Number of inpatient visits
* Number of emergency visits
* Number of outpatient visits
* Laboratory procedures
* Diagnoses
* Diabetes medications

---

## 🎯 Objective

The main objective is to build machine learning models that can identify patients who are at higher risk of **early hospital readmission (<30 days)**.

The project investigates:

1. Data preprocessing
2. Feature engineering
3. Categorical encoding
4. Class imbalance
5. Random Forest classification
6. XGBoost classification
7. Feature importance
8. Feature selection
9. Model evaluation
10. Gender and race subgroup analysis

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before training the models.

### Missing Values

Columns with more than 80% missing values were removed.

The following features were excluded:

* `weight`
* `max_glu_serum`
* `A1Cresult`

Patient and admission identifiers that were not used as predictive features were also removed, including:

```text
encounter_id
patient_nbr
admission_type_id
discharge_disposition_id
admission_source_id
```

---

## 🔧 Feature Engineering

### Age

The original age ranges were converted into numerical midpoint values.

For example:

```text
[20-30) → 25
[30-40) → 35
[40-50) → 45
```

This allows tree-based models to work directly with age as a numerical feature.

### Medication Features

Medication status was encoded numerically:

```text
No     → 0
Steady → 1
Up     → 2
Down   → -1
```

### Diagnosis Groups

The original diagnosis codes were transformed into broader groups using the first character of the diagnosis code.

For example, diagnosis features were transformed into:

```text
diag_1_group
diag_2_group
diag_3_group
```

This reduces the complexity of the original diagnosis codes while preserving broad diagnostic information.

### Categorical Encoding

Categorical variables were converted into numerical features using **one-hot encoding**.

After preprocessing and encoding, the dataset contained approximately:

**157 features**

---

## ⚖️ Class Imbalance

The target variable is imbalanced.

On the test set:

| Class | Meaning                       | Samples |
| ----- | ----------------------------- | ------: |
| 0     | Not readmitted within 30 days |  18,083 |
| 1     | Readmitted within 30 days     |   2,271 |

Because the positive class is much smaller, accuracy alone is not sufficient to evaluate the models.

To address class imbalance:

* Random Forest used `class_weight='balanced'`
* XGBoost used `scale_pos_weight`

Special attention was given to **precision, recall, and F1-score for class 1**.

---

## 🧪 Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

A stratified split was used to preserve the class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

# 🌲 Random Forest

Random Forest models were trained using:

```text
n_estimators = 200
class_weight = balanced
```

Two configurations were tested:

* `max_leaf_nodes = 100`
* `max_leaf_nodes = 300`

### Random Forest — 100 Leaf Nodes

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| 0     |      0.92 |   0.68 |     0.78 |
| 1     |      0.17 |   0.53 |     0.26 |

**Accuracy:** 66%

### Random Forest — 300 Leaf Nodes

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| 0     |      0.92 |   0.73 |     0.81 |
| 1     |      0.18 |   0.47 |     0.26 |

**Accuracy:** 70%

The results show the challenge of predicting the minority class: although the models achieve relatively high performance for class 0, precision for early readmission remains low.

---

# 🚀 XGBoost

XGBoost models were also trained and evaluated.

Two feature configurations were tested:

1. All available features
2. Top 20 features selected using Random Forest feature importance

Two tree configurations were also evaluated:

* `max_leaves = 100`
* `max_leaves = 300`

### XGBoost — All Features

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| 0     |      0.92 |   0.70 |     0.79 |
| 1     |      0.17 |   0.50 |     0.26 |

**Accuracy:** 67%

### XGBoost — Top 20 Features

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| 0     |      0.92 |   0.64 |     0.75 |
| 1     |      0.16 |   0.55 |     0.25 |

**Accuracy:** 63%

The top-20 feature experiment demonstrates how reducing the feature space affects model performance.

---

## ⭐ Feature Importance

The most important features identified by the Random Forest models included:

1. `number_inpatient`
2. `num_medications`
3. `num_lab_procedures`
4. `time_in_hospital`
5. `number_emergency`
6. `number_diagnoses`
7. `age`
8. `num_procedures`
9. `insulin`
10. `number_outpatient`
11. `metformin`
12. `diabetesMed`
13. `medical_specialty_Cardiology`
14. `glipizide`
15. `diag_1_group_7`
16. `diag_1_group_2`
17. `glyburide`
18. `race_Caucasian`
19. `gender_Male`
20. `diag_2_group_4`

Hospital utilization features such as previous inpatient and emergency visits were among the most influential features in the trained tree-based models.

---

# 🔍 Fairness Analysis

The project also investigates whether model recall differs across demographic groups.

The metric used here is **recall for class 1**, representing the proportion of patients who were actually readmitted within 30 days that the model identified.

## Gender

### XGBoost

| Gender | Recall |
| ------ | -----: |
| Female |   0.51 |
| Male   |   0.49 |

### Random Forest

| Gender | Recall |
| ------ | -----: |
| Female |   0.55 |
| Male   |   0.51 |

---

## Race

### XGBoost

| Race             | Recall |
| ---------------- | -----: |
| Caucasian        |   0.51 |
| African American |   0.48 |
| Hispanic         |   0.48 |
| Other            |   0.35 |
| Asian            |   0.40 |

### Random Forest

| Race             | Recall |
| ---------------- | -----: |
| Caucasian        |   0.54 |
| African American |   0.53 |
| Hispanic         |   0.56 |
| Other            |   0.50 |
| Asian            |   0.40 |

These results show differences in recall between demographic groups. However, subgroup recall alone is not sufficient to establish whether a model is systematically biased.

A more complete fairness analysis would also consider:

* Group sample sizes
* Confidence intervals
* False-positive rates
* False-negative rates
* Precision by subgroup
* Calibration
* Other fairness metrics

---

# 📈 Model Evaluation

The following evaluation methods were used:

* Classification Report
* Precision
* Recall
* F1-score
* Accuracy
* Confusion Matrix
* Feature Importance
* Demographic subgroup recall

Because the dataset is imbalanced, **class-1 recall and F1-score are particularly important** when interpreting the results.

---

# 🛠️ Technologies

The project was implemented using Python and the following libraries:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-readmission-prediction.git
cd diabetes-readmission-prediction
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost jupyter
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
diabetes_readmission.ipynb
```

Make sure the dataset is available at the expected path before running the notebook.

---

# 📁 Project Structure

```text
diabetes-readmission-prediction/
│
├── README.md
├── requirements.txt
├── diabetes_readmission.ipynb
│
├── data/
│   └── diabetic_data.csv
│
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── fairness_analysis.png
│
└── .gitignore
```

> The dataset may need to be excluded from the repository depending on its license and distribution requirements.

---

# ⚠️ Limitations

This project has several limitations:

* The target class is highly imbalanced.
* The model's precision for early readmission is relatively low.
* No cross-validation was performed in the current experiment.
* Hyperparameter optimization was limited.
* Fairness analysis is based primarily on subgroup recall.
* The diagnosis grouping strategy is relatively coarse.
* The project does not provide causal explanations for readmission.
* The model has not been clinically validated.
* Results from this historical dataset should not be assumed to generalize to modern hospitals or different populations.

---

# 🔮 Future Improvements

Potential improvements include:

* Hyperparameter optimization with `GridSearchCV`, `RandomizedSearchCV`, or Optuna
* Stratified cross-validation
* ROC-AUC and PR-AUC evaluation
* Calibration analysis
* SHAP-based model explainability
* More comprehensive fairness metrics
* Confidence intervals for subgroup metrics
* Better missing-value treatment
* Building a reusable Scikit-learn pipeline
* Experiment tracking
* Model deployment using FastAPI or Streamlit

---

# 👨‍💻 Project Goal

This project was developed as a machine learning portfolio project to practice:

**Data Cleaning → Feature Engineering → Classification → Imbalanced Learning → Model Evaluation → Feature Selection → Fairness Analysis**

It demonstrates an end-to-end machine learning workflow on a healthcare dataset.
