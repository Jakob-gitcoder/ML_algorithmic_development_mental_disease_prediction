
# Project Title: Detection of Anxiety and Distress in ICD Patients

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Summary](#dataset-summary)
3. [Machine Learning Models](#machine-learning-models)
4. [Evaluation Strategy](#evaluation-strategy)
5. [Data Preprocessing](#data-preprocessing)
   - [Dataset Characteristics](#dataset-characteristics)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Feature Selection Ensemble](#feature-selection-ensemble)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Ensemble Methods](#ensemble-methods)
8. [Final Model Evaluation](#final-model-evaluation)
9. [Explainable AI (XAI)](#explainable-ai-xai)
10. [Code Implementation](#code-implementation)
11. [Resources and Acknowledgments](#resources-and-acknowledgments)

---

### Project Overview
This project aims to detect anxiety and distress in patients with Implantable Cardioverter Defibrillators (ICD) using machine learning (ML) models trained on real-life patient data. The goal is to achieve accurate predictions, balancing clinical sensitivity and specificity for positive and negative cases.

---

### Dataset Summary
- **Data Source:** Real-life clinical data
- **Class Balance:** ~8-10% positive cases (anxiety/distress) vs. ~90% healthy patients.
- **Features:** 39 unique features, including patient-reported measures and demographic data.

---

### Machine Learning Models
Following a literature review focused on the medical domain, several models were selected for their robustness and variety, ideal for later ensemble modeling:
- Random Forest (RF)
- Support Vector Classifier (SVC)
- K-Nearest Neighbor (KNN)
- Multi-layer Perceptron (MLP)
- XGBoost, Naive Bayes (NB)
- Light Gradient Boosting (LGBM)
- Decision Tree Classifier (DTC)
- Logistic Regression (LGR)

---

### Evaluation Strategy
Given the clinical context, evaluation metrics are chosen based on the need for balanced performance on both classes:
- **Primary Metric:** F1-macro score, balancing precision and recall across classes.
- **Cross-validation:** Stratified cross-validation to ensure generalizability and prevent overfitting.

---

### Data Preprocessing

#### Dataset Characteristics
Key preprocessing challenges include:
- Missing values
- Small dataset size
- High feature-to-sample ratio
- Severe class imbalance

#### Preprocessing Pipeline
To tackle these issues, a comprehensive preprocessing pipeline includes:
1. **Imputation:** Simple Imputer, KNN Imputer, Iterative Imputer, Iterative Imputer with RandomForest.
2. **Scaling:** Standard Scaler, MinMax Scaler, Robust Scaler, Quantile Transformer.
3. **Feature Selection:** Various methods like Fisher Score, Mutual Information, Chi-squared, and recursive feature elimination.
4. **Balancing:** Random Under/Oversampler, SMOTEEEN, SMOTETomek.

*For a visual guide, refer to the Preprocessing Pipeline diagram:*
![Preprocessing Pipeline](./assets/Preprocessing_pipeline.jpg)

#### Feature Selection Ensemble
Combining filter, wrapper, and embedded methods, the feature selection ensemble enhances robustness by integrating statistical measures and dependency assessments.

---

### Hyperparameter Tuning
Optimizing hyperparameters iteratively through:
1. **RandomSearchCV:** Identifying promising parameter ranges.
2. **GridSearchCV:** Exhaustive testing within narrowed parameter ranges.
3. **Bayesian Optimization:** Adaptive approach using past iteration performance to converge towards optimal parameters.

---

### Ensemble Methods

**Stacking Classifier**: Combines base-learners of varying complexities to train a meta-learner, enhancing robustness through model diversity.

**Voting Classifier**: A simpler ensemble technique that averages predictions via hard/soft voting, balancing accuracy and computational efficiency.

---

### Final Model Evaluation
For final evaluations, multiple metrics are employed, including:
- Recall, Precision, F1 (per class)
- ROC-AUC, PRC-AUC, Accuracy

Each metric's standard deviation across cross-validation folds provides insight into the model's stability.

---

### Explainable AI (XAI)

#### Use Case
XAI techniques, specifically Shapley Additive Values (SHAP), provide interpretability, enabling insights into feature contributions and supporting clinician decisions.

#### Visualization Tools
1. **Global SHAP (Beeswarm Plot):** Illustrates feature importance across samples.
2. **Local SHAP (Force Plots):** Analyzes single predictions, highlighting feature influence on individual cases.

---

### Code Implementation

1. **Pipeline Preprocessing**
2. **Scaling**
3. **Tuned Models**
4. **Stacking/Voting Classifiers**
5. **Metrics for Final Models**
6. **SHAP Analysis**

Each file corresponds to a development phase, enabling straightforward reproducibility and understanding of the project's code structure.

---

### Resources and Acknowledgments
- **Data Source:** [Link or Citation]
- **External Libraries Used:** (e.g., `scikit-learn`, `xgboost`, `shap`)
- **Special Thanks to:** [Contributors, Mentors, etc.]

---
