# -*- coding: utf-8 -*-
"""

## Selected Dataset: Pima Indians Diabetes Database
**Link:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

## Project Workflow
```
diabetes.csv
    |
    v
Task 1: Data Loading
    |
    v
Task 2: Data Preprocessing (6 steps)
    |
    v
Task 3: Pipeline Creation
    |
    v
Task 4: Model Selection + Justification
    |
    v
Task 5: Model Training
    |
    v
Task 6: Cross-Validation
    |
    v
Task 7: Hyperparameter Tuning (GridSearchCV)
    |
    v
Task 8: Best Model Selection
    |
    v
Task 9: Performance Evaluation
    |
    v
Task 10: Save Model --> best_model.pkl
    |
    v
Task 11: Gradio Web App  <-- loads best_model.pkl
```

## Task 1: Data Loading (5 Marks)
Load the dataset, print shape, columns, first rows, and class distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

df = pd.read_csv('diabetes.csv')

df.head()

print('\nDataset shape')
print(df.shape)
print('\nDataset info')
df.info()

print('\nBasic Statistics')
print(df.describe().round(2))

# distribution
print('Target Variable: Outcome')
print(df['Outcome'].value_counts())
print(df['Outcome'].value_counts(normalize=True).round(3))

"""## Task 2: Data Preprocessing (10 Marks)
At least 5 distinct preprocessing steps documented below.
"""

# STEP 1- Replace Biologically Impossible Zeros with NaN
# Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be 0

df_clean = df.copy()
zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print('STEP 1: Zero counts in medically impossible columns')
for col in zero_invalid_cols:
    n = (df_clean[col] == 0).sum()
    print(f'{col:<20}: {n} zeros')

df_clean[zero_invalid_cols] = df_clean[zero_invalid_cols].replace(0, np.nan)
print()
print('Zeros replaced with NaN for proper imputation.')

# STEP 2 — Handle Missing Values (median)
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0].to_string())
print()

for col in zero_invalid_cols:
    med = df_clean[col].median()
    df_clean[col].fillna(med, inplace=True)

print(f'\nTotal missing values after imputation: {df_clean.isnull().sum().sum()}')

# STEP 3 — Outlier Detection and Capping (IQR / Winsorization)
features_all = df_clean.drop('Outcome', axis=1).columns.tolist()

outlier_counts = {}
for col in features_all:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    outlier_counts[col] = n_out
    df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)

# Before/After boxplots
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(features_all):
    axes[i].boxplot([df[col], df_clean[col]], labels=['Original', 'Capped'])
    axes[i].set_title(col, fontsize=10)
plt.suptitle('Boxplots: Original vs After Outlier Capping', fontsize=13)
plt.tight_layout()
plt.show()

# Step 4-Feature Engineering

print('STEP 4: Feature Engineering')

# BMI Category (clinical classification)
df_clean['BMI_Category'] = pd.cut(
    df_clean['BMI'],
    bins=[0, 18.5, 24.9, 29.9, 100],
    labels=[0, 1, 2, 3]
).astype(int)

# Glucose-to-Insulin Ratio (proxy for insulin resistance)
df_clean['Glucose_Insulin_Ratio'] = df_clean['Glucose'] / (df_clean['Insulin'] + 1)

# Age Group (clinical risk bands)
df_clean['Age_Group'] = pd.cut(
    df_clean['Age'],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
).astype(int)

# New features created:
# BMI_Category          : 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
# Glucose_Insulin_Ratio : Glucose / (Insulin + 1)  — insulin resistance
# Age_Group             : 0=<30, 1=30-45, 2=45-60, 3=60+
print(f'Dataset shape after engineering: {df_clean.shape}')

# Step 5—Correlation Analysis

corr = df_clean.corr()
plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap (After Preprocessing)', fontsize=13)
plt.tight_layout()
plt.show()

print('Correlation with Outcome (sorted):')
print(corr['Outcome'].drop('Outcome').sort_values(ascending=False).round(3).to_string())

# STEP 6 — Stratified Train-Test Split
from sklearn.model_selection import train_test_split

X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Training set : {X_train.shape[0]} samples')
print(f'Test set : {X_test.shape[0]} samples')
print()
print(f'Feature columns({len(X.columns)}): {X.columns.tolist()}')

"""## Task 3: Pipeline Creation (10 Marks)
Construct a standard ML pipeline integrating preprocessing and the model.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Pipeline: SimpleImputer -> StandardScaler -> RandomForestClassifier
pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler()),
    ('classifier',RandomForestClassifier(random_state=42))
])

print('ML Pipeline')
print(pipeline)

"""## Task 4: Primary Model Selection (5 Marks)
Choose a suitable algorithm and justify the choice.
"""

print('=== Primary Model: Random Forest Classifier ===')
print()
print("""JUSTIFICATION:

1. NON-LINEAR RELATIONSHIPS
   Diabetes involves complex feature interactions (e.g., high Glucose + high
   BMI + Age > 45). Random Forest captures these through ensemble trees.

2. ROBUSTNESS TO OUTLIERS
   Tree splits are rank-based — unaffected by skewed medical distributions.

3. BUILT-IN FEATURE IMPORTANCE
   Provides clinical interpretability — which factors drive diabetes risk.

4. HANDLES CLASS IMBALANCE
   class_weight='balanced' compensates for 65/35 imbalance during tuning.

5. ENSEMBLE STABILITY
   Averaging 100+ trees reduces variance vs a single decision tree.

6. NO MANDATORY FEATURE SCALING
   Inherently scale-invariant (scaling is still included in pipeline for
   consistency across all models in the grid search).

Alternatives Rejected:
  - Logistic Regression : Assumes linear decision boundary (too restrictive)
  - SVM                 : Slow on large grids; less interpretable
  - KNN                 : Sensitive to irrelevant features; slow inference
  - Naive Bayes         : Independence assumption violated (Glucose & Insulin correlated)
""")

"""## Task 5: Model Training (10 Marks)
Train the pipeline on the training set.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train
pipeline.fit(X_train, y_train)
print('Pipeline trained successfully\n')

y_train_pred = pipeline.predict(X_train)
y_test_pred_base = pipeline.predict(X_test)

print(f'Training Accuracy :{accuracy_score(y_train, y_train_pred):.4f}')
print(f'Test Accuracy : {accuracy_score(y_test, y_test_pred_base):.4f}')
print()
print('Test Set Report')
print(classification_report(y_test, y_test_pred_base,target_names=['Non-Diabetic', 'Diabetic']))

# Confusion Matrix
cm_base = confusion_matrix(y_test, y_test_pred_base)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Random Forest — Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature Importances
rf_base   = pipeline.named_steps['classifier']
feat_imp  = pd.Series(rf_base.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importances — Baseline Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

"""## Task 6: Cross-Validation (10 Marks)
Apply 10-fold Stratified Cross-Validation and report average score with standard deviation.
"""

from sklearn.model_selection import StratifiedKFold, cross_validate

print('10-Fold Stratified Cross-Validation')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X, y,
    cv=skf,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
    n_jobs=-1
)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
print(f'{"Metric":<12} {"Test Mean":>12} {"Test Std":>10} {"Train Mean":>12}')
for m in metrics:
    tm  = cv_results[f'test_{m}'].mean()
    ts  = cv_results[f'test_{m}'].std()
    trm = cv_results[f'train_{m}'].mean()
    print(f'{m:<12} {tm:>12.4f} {ts:>10.4f} {trm:>12.4f}')

print()
print('Per-Fold Accuracy:')
for i, acc in enumerate(cv_results['test_accuracy'], 1):
    print(f'Fold {i:2d}: {acc:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

folds = range(1, 11)
axes[0].plot(folds, cv_results['test_accuracy'],  'o-', label='Test',  color='steelblue')
axes[0].plot(folds, cv_results['train_accuracy'], 's--', label='Train', color='darkorange')
axes[0].axhline(cv_results['test_accuracy'].mean(), color='red', linestyle=':',
                label=f"Mean = {cv_results['test_accuracy'].mean():.3f}")
axes[0].set_title('CV Accuracy per Fold')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].set_xticks(list(folds))

test_means = [cv_results[f'test_{m}'].mean() for m in metrics]
bars = axes[1].barh(metrics, test_means, color=sns.color_palette('viridis', len(metrics)))
for i, v in enumerate(test_means):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
axes[1].set_title('Average CV Scores (10-Fold)')
axes[1].set_xlabel('Score')
axes[1].set_xlim(0, 1.12)

plt.suptitle('10-Fold Stratified Cross-Validation Results', fontsize=13)
plt.tight_layout()
plt.show()

"""## Task 7: Hyperparameter Tuning (10 Marks)
Optimize the model using GridSearchCV- print parameters tested and best results.
"""

from sklearn.model_selection import GridSearchCV

print('GridSearchCV — Hyperparameter Tuning')
print()

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth' : [None, 5, 10, 15],
    'classifier__min_samples_split' : [2, 5, 10],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight' : [None, 'balanced']
}

print('Parameter Grid:')
for k, v in param_grid.items():
    print(f'{k}: {v}')
# \nTotal combinations: 3x4x3x2x2 = 144  x  5 folds = 720 fits
print('Scoring metric: ROC-AUC (best for imbalanced medical classification)')

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print()
print('Running GridSearchCV...')
grid_search.fit(X_train, y_train)

print()
print(f'Best ROC-AUC(CV):{grid_search.best_score_:.4f}')
print('Best Hyperparameters:')
for p, v in grid_search.best_params_.items():
    print(f'{p}:{v}')

cv_df = pd.DataFrame(grid_search.cv_results_)
top10 = cv_df.sort_values('mean_test_score', ascending=False).head(10)

cols = [
    'param_classifier__n_estimators',
    'param_classifier__max_depth',
    'param_classifier__min_samples_split',
    'param_classifier__max_features',
    'param_classifier__class_weight',
    'mean_test_score', 'std_test_score'
]
print('Top 10 Configurations by ROC-AUC:')
print(top10[cols].round(4).reset_index(drop=True))

"""## Task 8: Best Model Selection (10 Marks)
Select the final best-performing model from hyperparameter tuning.
"""

from sklearn.metrics import roc_auc_score

best_model = grid_search.best_estimator_

print('Best Model')
print(best_model)
print()
print('Best Parameters:')
for p, v in grid_search.best_params_.items():
    print(f'  {p.replace("classifier-", ""):}: {v}')

print(f'\nBest CV ROC-AUC : {grid_search.best_score_:.4f}')
print()

# Compare baseline vs tuned
y_base_prob  = pipeline.predict_proba(X_test)[:, 1]
y_tuned_pred = best_model.predict(X_test)
y_tuned_prob = best_model.predict_proba(X_test)[:, 1]

print(f'{"Metric":<15} {"Baseline":>12} {"Tuned":>12}')
print(f'{"Accuracy":<15} {accuracy_score(y_test, pipeline.predict(X_test)):>12.4f} {accuracy_score(y_test, y_tuned_pred):>12.4f}')
print(f'{"ROC-AUC":<15} {roc_auc_score(y_test, y_base_prob):>12.4f} {roc_auc_score(y_test, y_tuned_prob):>12.4f}')
print()
print('The tuned best_model is selected as the FINAL model.')

"""## Task 9: Model Performance Evaluation (10 Marks)
Comprehensive evaluation of the best model on the held-out test set.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
ap   = average_precision_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print('   FINAL MODEL — TEST SET EVALUATION')
print(f'  Accuracy : {acc:.4f}')
print(f'  Precision : {prec:.4f}')
print(f'  Recall (Sensitivity) : {rec:.4f}')
print(f'  F1-Score : {f1:.4f}')
print(f'  ROC-AUC : {auc:.4f}')
print(f'  Average Precision : {ap:.4f}')
print(f'  Specificity: {tn/(tn+fp):.4f}')
print()
print(f'  Confusion Matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print()
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1, label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

# Precision-Recall Curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
axes[2].plot(rec_vals, prec_vals, color='darkorange', lw=2, label=f'AP = {ap:.3f}')
axes[2].axhline(y_test.mean(), color='gray', linestyle='--', label='Baseline')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve')
axes[2].legend()

plt.suptitle('Best Model —Evaluation', fontsize=13)
plt.tight_layout()
plt.show()

# Feature importances
rf_tuned      = best_model.named_steps['classifier']
feat_imp_tuned = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=feat_imp_tuned.values, y=feat_imp_tuned.index, palette='magma')
plt.title('Tuned Random Forest — Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print('Feature Importances (Tuned Model):')
print(feat_imp_tuned.round(4).to_string())

"""## Task 10: Save the Trained Model as a .pkl File

After completing all preprocessing, training, and tuning — serialize the best model pipeline and supporting metadata using `joblib` into `best_model.pkl`.  
This file is then used directly by the Gradio app **without re-running any training code**.
"""

import pickle
import os

# Pack everything the Gradio app will need
model_payload = {
    'model'        : best_model,
    'feature_cols' : list(X.columns),
    'class_names'  : ['Non-Diabetic', 'Diabetic'],
    'best_params'  : grid_search.best_params_,
    'cv_roc_auc'   : round(grid_search.best_score_, 4),
    'test_accuracy': round(acc, 4),
    'test_roc_auc' : round(auc, 4),
}

pkl_path = 'best_model.pkl'

# Save pickle
with open(pkl_path, 'wb') as f:
    pickle.dump(model_payload, f)

size_kb = os.path.getsize(pkl_path) / 1024

# Verify:
print('Verifying saved best_model.pkl')
print()

with open('best_model.pkl', 'rb') as f:
    loaded_payload = pickle.load(f)

loaded_model = loaded_payload['model']
loaded_cols  = loaded_payload['feature_cols']

print(f'Model type loaded : {type(loaded_model).__name__}')
print(f'Feature columns : {loaded_cols}')
print()

# Run same test-set row through both models
sample      = X_test.iloc[[0]]
pred_orig   = best_model.predict(sample)[0]
pred_loaded = loaded_model.predict(sample)[0]
prob_loaded = loaded_model.predict_proba(sample)[0]

print('Sample input (first test row):')
print(sample.to_string())

if pred_orig == pred_loaded:
    print('Verification PASSED')
else:
    print('WARNING: mismatch detected!')

"""## Task 11: Gradio Web Interface — Loads from best_model.pkl (10 Marks)

The Gradio app:
1. **Loads** `best_model.pkl` at startup (no model code / training needed here)
2. Accepts all 8 original patient inputs via interactive widgets
3. Internally computes the 3 engineered features (BMI_Category, Glucose_Insulin_Ratio, Age_Group)
4. Passes the full 11-feature vector through the loaded pipeline
5. prints prediction label, confidence bar, and a risk summary
"""