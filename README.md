# Student Depression Behavioral Analysis
 
## Research Objective 

This project investigates behavioral and academic factors associated with student depression using:
* Exploratory Data Analysis (EDA)
* Statistical hypothesis testing
* Machine learning classification (Logistic Regression)
* Cross-validation for generalization reliability

The goal is to produce a reproducible, research-style data analysis pipeline suitable for academic or clinical research environments.

---

## Research Questions

1. Is sleep duration significantly associated with depression status?
2. Do academic or financial stress levels correlate with higher depression risk?
3. Which behavioral factors show the strongest statistical relationship with depression?

---

## Dataset

* **Rows:** 27,901 students
* **Features:** 18 behavioral & demographic variables
* **Target:** `depression` (binary)

**Key predictors used in ML:**
* `sleep_duration`
* `age`
* `cgpa`
* `academic_pressure`
* `work_study_hours`

---

## Research Methodology & Machine Learning Pipeline

### 1. Handling Missing Data: Multiple Imputation (MICE)
A significant challenge in the dataset was the missingness in the `sleep_duration` feature (approx. 48.55%). Initially, dropping these records (Listwise Deletion) would have severely reduced the statistical power and potentially introduced selection bias (MAR/MNAR). 

To build a robust, research-grade model, I implemented **Multiple Imputation by Chained Equations (MICE)** using scikit-learn's `IterativeImputer`. This allowed the model to retain the entire dataset (N = 27,901) by intelligently estimating missing values based on the covariance of other clinical and academic features.

### 2. Zero-Leakage Cross-Validation Architecture
To strictly prevent data leakage, the entire preprocessing logic was encapsulated within a `scikit-learn Pipeline`. 

The execution flow for each Stratified K-Fold is as follows:
1. **Imputation:** `IterativeImputer` fits only on the training fold, estimating missing `sleep_duration` values.
2. **Scaling:** `RobustScaler` reduces the influence of extreme outliers (e.g., extreme age or work/study hours).
3. **Classification:** A `LogisticRegression` (liblinear solver) acts as the baseline interpretable classifier.

**Results:** This robust pipeline achieved a highly stable Cross-Validation Mean AUC of **0.818** across the full population of 27,901 students.

---

## Data Quality & Missing Data Handling

### Missing Data Summary
| Feature | Missing % | Strategy Applied |
|---------|-----------|------------------|
| sleep_duration | 48.55% | Multiple Imputation (MICE) |
| age | 0.1% | Multiple Imputation (MICE) |
| cgpa | 2.3% | Multiple Imputation (MICE) |
| academic_pressure | 0% | None required |
| work_study_hours | 3.1% | Multiple Imputation (MICE) |

* **Total cases used in ML pipeline:** 27,901 / 27,901 (100%)
* **Rows dropped:** 0

**Missing data analysis:** While complete-case filtering would have reduced the effective sample size by ~48% and introduced severe selection bias, the implementation of MICE successfully preserved the entire dataset's statistical power and original prevalence distribution.

--- 

## Feature Selection

Five behavioral and academic features were selected based on:
1. **Theoretical foundation**: Established risk factors in mental health literature
2. **Interpretability**: Measurable in real screening contexts
3. **Statistical independence**: Variance Inflation Factor (VIF) < 5

**Feature correlations**: Maximum pairwise correlation = 0.42 (academic_pressure vs work_study_hours), indicating low multicollinearity.

---

## Statistical Power

With **n = 27,901** (full sample retained via MICE) and a depression prevalence of **58.55%**:
- Post-hoc power analysis suggests >99.9% statistical power to detect even very small effects at α=0.05 under the current sample size.
- **Conclusion:** The MICE-imputed sample provides exceptional statistical power to detect small-to-moderate behavioral associations with high confidence.

---

## Exploratory Data Analysis Results

The following visualizations summarize key behavioral and lifestyle patterns associated with student depression.

### Depression Distribution
![Depression Distribution](artifacts/depression_distribution.png)

This chart shows the prevalence of depression within the dataset, providing a baseline understanding of class balance for further analysis.

---

### Age Distribution
![Age Distribution](artifacts/age_distribution.png)

The histogram illustrates the demographic age structure of the student population, which may influence mental health risk patterns.

---

### Sleep Duration vs Depression
![Sleep vs Depression](artifacts/sleep_vs_depression.png)

This visualization explores whether reduced sleep duration is associated with higher depression prevalence, a commonly reported psychological factor.

---

### Feature Correlation Heatmap
![Correlation Heatmap](artifacts/correlation_heatmap.png)

The heatmap highlights statistical relationships among numeric variables, helping identify potential predictors and confounding factors related to depression.

---

# Statistical Inference

## Sleep Duration and Depression

A Mann–Whitney U test revealed a statistically significant difference in sleep duration between students with and without depression (**p < 0.001**).

Students with depression slept fewer hours on average (**M = 5.82**) compared to non-depressed students (**M = 6.51**), with a small-to-moderate effect size (**Cohen’s d = 0.28**).

This finding suggests that **reduced sleep duration is meaningfully associated with depression risk** in the student population. Although the effect size is small-to-moderate, the consistent population-level difference in sleep duration indicates a meaningful behavioral risk signal relevant for early mental-health screening.

---

## Machine Learning Pipeline

We implement a reproducible sklearn Pipeline:
* `RobustScaler` → handles outliers safely
* `LogisticRegression` (liblinear, max_iter=5000)
* Stratified train/test split
* ROC-AUC + accuracy + confusion matrix

**Why Pipeline?**
* Prevents data leakage
* Ensures reproducibility
* Mirrors real research deployment workflow

---

## Model Performance

The logistic regression classifier achieved:
- **ROC-AUC:** ≈ 0.815
- **Accuracy:** ≈ 0.751

These results indicate **moderate-to-strong discriminative ability** in identifying depression-associated behavioral patterns.

---

### Cross-Validation Performance

To evaluate generalization reliability without data leakage, we conducted a 5-fold stratified cross-validation. The MICE imputation and scaling were strictly fitted *inside* the CV loops.

**Data retained after Pipeline processing:** * Total rows used: **27,901 / 27,901** (100%)  
* Rows dropped: **0** * Overall depression prevalence: **58.55%**

**Cross-validation AUC scores (5 folds):**
* Fold 1: 0.8266
* Fold 2: 0.8189
* Fold 3: 0.8182
* Fold 4: 0.8144
* Fold 5: 0.8138

**Mean AUC:** 0.8184  
**Standard deviation:** 0.0051

The extremely tight spread of AUC scores proves that the predictive signal is robust, not overfitted, and independent of random train/test splits.

---

## Model Interpretability

Logistic regression coefficients (standardized) indicate:

| Feature | Interpretation / Directionality |
|---------|--------------------------------|
| Sleep duration | Protective factor: More sleep is associated with lower depression odds |
| Academic pressure | Risk factor: High pressure strongly increases depression odds |
| CGPA | Weak protective effect |

These interpretable relationships provide behaviorally meaningful insights and support potential use in early-risk screening frameworks.

**Note**: For future complex models (ensemble methods, neural networks), interpretability tools like SHAP will be implemented to maintain clinical explainability while capturing nonlinear patterns.

---

## Research Significance

This study demonstrates a reproducible behavioral risk modeling framework for identifying depression-associated patterns in student populations.

Key implications include:
* Potential support for early mental-health risk screening.
* Quantitative evidence linking sleep and academic stress to depression outcomes.
* A robust methodological demonstration of handling massive missing data (48.55%) via Multiple Imputation without sacrificing model integrity.

---

## Limitations

Several limitations should be considered when interpreting the findings:
1. **Cross-sectional Data:** The dataset is cross-sectional, preventing causal inference between behavioral factors and depression outcomes.
2. **Self-reported Metrics:** Depression status and behavioral variables are self-reported, which may introduce reporting bias.
3. **Imputation Dependency:** While missing data was robustly handled via MICE, retaining the missing `sleep_duration` records relies on the assumption that the data is Missing at Random (MAR) and can be estimated via other features.
4. **Linearity Assumption:** The logistic regression model captures linear relationships. More complex, nonlinear behavioral interactions might be missed.

---

## Future Work

Future research directions include:
* Incorporating longitudinal mental-health data to evaluate causal relationships.
* Expanding behavioral features such as social support, lifestyle habits, and digital activity.
* Exploring nonlinear models (e.g., tree-based ensembles or neural networks) **with SHAP-based interpretability to maintain clinical relevance**.
* Validating the model on independent student populations to assess real-world generalization.

---

## Reproducibility

To reproduce the full analysis pipeline:

### Environment Setup
Python version: **3.10+**

Install dependencies:
```bash
pip install -r requirements.txt
python -m src.main