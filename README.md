# Student Depression Behavioral Analysis
 
## Research Objective 

This project investigates behavioral and academic factors associated with student depression using:

Exploratory Data Analysis (EDA)

Statistical hypothesis testing

Machine learning classification (Logistic Regression)

Cross-validation for generalization reliability

The goal is to produce a reproducible, research-style data analysis pipeline suitable for academic or clinical research environments.
---

## Research Questions

1. Is sleep duration significantly associated with depression status?
2. Do academic or financial stress levels correlate with higher depression risk?
3. Which behavioral factors show the strongest statistical relationship with depression?

---

## Dataset

Rows: 27,901 students

Features: 18 behavioral & demographic variables

Target: depression (binary)

Key predictors used in ML:

sleep_duration

age

cgpa

academic_pressure

work_study_hours

---

## Data Quality & Missing Data Handling

### Missing Data Summary
| Feature | Missing % | Strategy |
|---------|-----------|----------|
| sleep_duration | 48.55% | Listwise deletion (MCAR not rejected; χ²=2.8, p=0.42)|
| age | 0.1% | Listwise deletion |
| cgpa | 2.3% | Listwise deletion |
| academic_pressure | 0% | None required |
| work_study_hours | 3.1% | Listwise deletion |

**Total cases used:** 14,354 / 27,901 (51.45%)

A total of 13,547 observations were excluded due to missing values 
in key predictors (primarily sleep_duration, 48.55% missing).

**Missing data analysis**: A chi-square test did not reject the MCAR assumption (χ²=2.8, p=0.42).
However, given the high missing proportion in sleep_duration (48.55%),
complete-case filtering substantially reduced the effective sample size
and may introduce selection bias.

Therefore, while listwise deletion was applied for analytical consistency,
results should be interpreted with caution.
--- 

## Feature Selection

Five behavioral and academic features were selected based on:

1. **Theoretical foundation**: Established risk factors in mental health literature
2. **Data quality**: All features had <20% missing data
3. **Interpretability**: Measurable in real screening contexts
4. **Statistical independence**: Variance Inflation Factor (VIF) < 5

**Feature correlations**: Maximum pairwise correlation = 0.42
(academic_pressure vs work_study_hours), indicating low multicollinearity.

---


## Statistical Power

With n=14,354 (complete-case sample) and depression prevalence of 58.8%:

- Post-hoc power analysis suggests >99% statistical power
to detect small effects (d=0.2) at α=0.05 under current sample size.
- Minimum detectable effect: d=0.06 with 80% power
- Conclusion: The retained sample remains sufficiently powered 
  to detect small-to-moderate behavioral associations.

---

## EDA includes:

- Dataset integrity checks and column normalization
- Distribution analysis of demographic and behavioral variables
- Visualization of depression prevalence across factors
- Correlation analysis among numeric features

All visual outputs are saved for reproducible research reporting.

---

## Exploratory Data Analysis Results

The following visualizations summarize key behavioral and lifestyle
patterns associated with student depression.

### Depression Distribution

![Depression Distribution](artifacts/depression_distribution.png)

This chart shows the prevalence of depression within the dataset,
providing a baseline understanding of class balance for further analysis.

---

### Age Distribution
![Age Distribution](artifacts/age_distribution.png)

The histogram illustrates the demographic age structure of the student
population, which may influence mental health risk patterns.

---

### Sleep Duration vs Depression
![Sleep vs Depression](artifacts/sleep_vs_depression.png)

This visualization explores whether reduced sleep duration is associated
with higher depression prevalence, a commonly reported psychological factor.

---

### Feature Correlation Heatmap
![Correlation Heatmap](artifacts/correlation_heatmap.png)

The heatmap highlights statistical relationships among numeric variables,
helping identify potential predictors and confounding factors related to depression.

---

# Statistical Inference

## Sleep Duration and Depression

A Mann–Whitney U test revealed a statistically significant difference in sleep duration between students with and without depression  
(**p = 6.34 × 10⁻⁶⁰**).

Students with depression slept fewer hours on average  
(**M = 5.82**) compared to non-depressed students (**M = 6.51**),  
with a **small-to-moderate effect size** (**Cohen’s d = 0.28**).

This finding suggests that **reduced sleep duration is meaningfully associated with depression risk** in the student population.

Although the effect size is small-to-moderate, the consistent population-level difference in sleep duration
may indicate a meaningful behavioral risk signal relevant for early mental-health screening.

---

## Machine Learning Pipeline

We implement a reproducible sklearn Pipeline:

RobustScaler → handles outliers safely

LogisticRegression (liblinear, max_iter=5000)

Stratified train/test split

ROC-AUC + accuracy + confusion matrix

Why Pipeline?

Prevents data leakage

Ensures reproducibility

Mirrors real research deployment workflow

---

## Model Performance

The logistic regression classifier achieved:

- ROC-AUC ≈ 0.83  
- Accuracy ≈ 0.76  
- Cross-validation mean AUC (0.828 ± 0.006)
indicates low variance across folds,
suggesting stable generalization performance.

These results indicate **moderate discriminative ability**
in identifying depression-associated behavioral patterns.

Importantly, consistent cross-validation performance
suggests the model is **not overfitting**
and may generalize to similar student populations.

---

## Model Interpretability

Logistic regression coefficients (standardized) indicate:

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|------------|------------|----------------|
| Sleep duration | -0.42 | 0.66 | Each additional hour of sleep reduces depression odds by 34% |
| Academic pressure | +0.38 | 1.46 | High academic pressure increases depression odds by 46% |
| CGPA | -0.15 | 0.86 | Higher CGPA shows weak protective effect |

These interpretable relationships provide behaviorally meaningful insights
and support potential use in early-risk screening frameworks.

**Note**: For future complex models (ensemble methods, neural networks),
interpretability tools like SHAP would be implemented to maintain
clinical explainability while capturing nonlinear patterns.

---

### Cross-Validation Performance

To evaluate generalization reliability, we conducted 5-fold stratified cross-validation 
using the same LogisticRegression + RobustScaler pipeline.

**Data retained after cleaning:**  
- Total rows used: **14,354 / 27,901** (51.45%)  
- Rows dropped due to missing data: **13,547**  
- Overall depression prevalence after filtering: **58.8%**

**Cross-validation AUC scores (5 folds):**
- Fold 1: 0.8246  
- Fold 2: 0.8300  
- Fold 3: 0.8335  
- Fold 4: 0.8330  
- Fold 5: 0.8183  

**Mean AUC:** 0.8279  
**Standard deviation:** 0.0064  

The low variance across folds indicates strong model stability and 
consistent discriminative performance across resampled subsets of the data.

These results support the model's robustness and suggest that 
performance is not driven by random train/test splits.

---

## Research Significance

This study demonstrates a reproducible behavioral risk modeling framework
for identifying depression-associated patterns in student populations.

Key implications include:

Potential support for early mental-health risk screening

Quantitative evidence linking sleep and academic stress to depression outcomes

A foundation for future clinical or longitudinal research

By combining statistical inference, machine learning prediction,
and reproducible analysis design,
this project reflects a research-oriented data science workflow
suitable for academic and applied mental-health research settings.

This framework demonstrates how classical statistical inference
and interpretable machine learning can be integrated
into a transparent behavioral risk modeling workflow.

---

## Limitations

Several limitations should be considered when interpreting the findings:

- The dataset is **cross-sectional**, preventing causal inference
  between behavioral factors and depression outcomes.

- Depression status and behavioral variables are **self-reported**,
  which may introduce reporting bias.

- A large proportion of **sleep duration values were missing**,
  potentially affecting statistical power and model generalization.

- The logistic regression model provides **moderate predictive ability**
  and may not capture complex nonlinear behavioral interactions.

These limitations highlight the need for
longitudinal data collection, richer clinical variables,
and more advanced modeling approaches in future research.

---

## Future Work

Future research directions include:

- Incorporating longitudinal mental-health data to evaluate causal relationships  
- Expanding behavioral features such as social support, lifestyle habits, and digital activity  
- Exploring nonlinear models (e.g., tree-based ensembles or neural networks)  
  **with SHAP-based interpretability to maintain clinical relevance**
- Validating the model on independent student populations  
  to assess real-world generalization

**Note on Interpretability**: While logistic regression provides direct coefficient 
interpretation, future complex models would benefit from SHAP (SHapley Additive 
explanations) to maintain explainability for clinical decision support.

---

## Reproducibility

To reproduce the full analysis pipeline:

### Environment Setup

Python version: **3.10+**

Install dependencies:

```bash
pip install -r requirements.txt

```

python -m src.main


--- 
## Academic Use & Citation

This repository is designed for educational and research demonstration
purposes in behavioral data analysis and mental-health risk modeling.

If this project contributes to academic work, please cite as:

Student Depression Behavioral Analysis, 2026.

---

