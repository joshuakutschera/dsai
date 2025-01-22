import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings for clarity

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, recall_score, roc_auc_score, roc_curve
)

from math import sqrt
from IPython.display import display

# Setting a universal plotting style for seaborn
sns.set(style="whitegrid", context="notebook")


# ## 1. Data Loading
# ### 1.1 Load the Dataset

def load_heart_data(path: str) -> pd.DataFrame:

    data = pd.read_csv(path)
    return data


heart_data = load_heart_data("Data/heart.csv")

# Basic preview
print("First 5 rows of the dataset:")
display(heart_data.head())

# Basic info
print("\nDataset Info:")
heart_data.info()

# Statistical description
print("\nDataset Description:")
display(heart_data.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(heart_data.isnull().sum())

# ## 4. Exploratory Data Analysis (EDA)
# ### 4.1 Univariate Analysis

def plot_numeric_distribution(df: pd.DataFrame, column: str, color: str = 'blue') -> None:
    plt.figure(figsize=(10, 5))
    mean_val = df[column].mean()
    median_val = df[column].median()

    sns.histplot(df[column], bins=30, kde=True, color=color, alpha=0.6)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# Numeric columns of interest
numeric_cols = ["age", "trtbps", "chol", "thalach", "oldpeak"]

for col in numeric_cols:
    plot_numeric_distribution(heart_data, col)

# Count plots for categorical columns
cat_cols = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]

plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 4, i)
    sns.countplot(data=heart_data, x=col, color='skyblue', alpha=0.7)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
plt.show()

# Distribution of target variable
plt.figure(figsize=(5, 4))
sns.countplot(data=heart_data, x='output', palette='Set2')
plt.title("Heart Attack Risk Distribution (Target)")
plt.xlabel("Output (0 = No Risk, 1 = Risk)")
plt.ylabel("Count")
plt.show()


# ### 4.2 Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = heart_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap for Numeric Features + Target")
plt.show()
# ### 4.3 Bivariate Analysis

# 1) Age vs. Maximum Heart Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='thalach', data=heart_data, hue='output', alpha=0.7, palette='Set1')
sns.regplot(x='age', y='thalach', data=heart_data, scatter=False, color='black', ci=None)
corr_age_thalach = heart_data['age'].corr(heart_data['thalach'])
plt.title(f"Age vs. Maximum Heart Rate (corr = {corr_age_thalach:.2f})")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalach)")
plt.legend(title="Output")
plt.show()

# 2) Chest Pain Type vs. Heart Attack Risk
plt.figure(figsize=(7, 5))
sns.countplot(data=heart_data, x='cp', hue='output', palette='viridis')
plt.title("Chest Pain Type vs. Heart Attack Risk")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.legend(title="Risk (output)")
plt.show()

# 3) Resting BP vs. Cholesterol, colored by output
plt.figure(figsize=(8, 5))
sns.scatterplot(x='trtbps', y='chol', hue='output', data=heart_data, palette='plasma', alpha=0.8)
plt.title("Resting Blood Pressure vs. Cholesterol by Heart Attack Risk")
plt.xlabel("Resting Blood Pressure (trtbps)")
plt.ylabel("Cholesterol (chol)")
plt.legend(title="Risk (output)")
plt.show()

# ## 5. Data Preparation
# ### 5.1 Feature/Target Separation
X = heart_data.drop(columns="output")
y = heart_data["output"]

# ### 5.2 Identify Numeric and Categorical Columns
numeric_features = ["age", "trtbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]

# ### 5.3 Preprocessing Pipelines
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# ### 5.4 Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ensures balanced distribution of the target
)

print(f"Training set size: {X_train.shape[0]} rows, Test set size: {X_test.shape[0]} rows")

# ## 6. Model Building
# 1. Decision Tree
# 2. Random Forest
# 3. Logistic Regression
#
# performance comparison. cross-validation.

def evaluate_model(model_name: str, y_true, y_pred) -> None:
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate additional metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else None

    # Print metrics
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"ROC AUC:    {auc if auc else 'N/A'}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


# Create pipelines for each model
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("dt", DecisionTreeClassifier(random_state=42))
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("lr", LogisticRegression(max_iter=2000, random_state=42))
])

# Train the models
dt_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

# Store them in a dict for convenience
models = {
    "Decision Tree": dt_pipeline,
    "Random Forest": rf_pipeline,
    "Logistic Regression": lr_pipeline
}

# Evaluate on test set
for model_name, model_pipeline in models.items():
    preds = model_pipeline.predict(X_test)
    evaluate_model(model_name, y_test, preds)

# ## 7. Cross-Validation

def cross_validate_model(model_pipeline, X, y, model_name: str, cv_folds=5) -> None:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"\n{model_name} - {cv_folds}-Fold Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} | Std Dev: {cv_scores.std():.4f}")


for model_name, model_pipeline in models.items():
    cross_validate_model(model_pipeline, X, y, model_name, cv_folds=5)

# ## 8. Feature Importance and Interpretation

# ### 8.1 Random Forest Feature Importances
rf_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_model_full.fit(X, y)

# Extract the underlying RandomForest model
rf_estimator = rf_model_full.named_steps['rf']

# Get the one-hot-encoded feature names
encoded_cat_cols = (
    rf_model_full.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(categorical_features)
)

all_feature_names = numeric_features + list(encoded_cat_cols)
importances = rf_estimator.feature_importances_

feat_imp_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis")
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

display(feat_imp_df)

# ### 8.2 Logistic Regression Coefficients
lr_model_full = Pipeline([
    ("preprocessor", preprocessor),
    ("lr", LogisticRegression(max_iter=2000, random_state=42))
])
lr_model_full.fit(X, y)

lr_estimator = lr_model_full.named_steps['lr']
coeffs = lr_estimator.coef_[0]

lr_coef_df = pd.DataFrame({
    "feature": all_feature_names,
    "coefficient": coeffs
}).sort_values(by="coefficient", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=lr_coef_df, x="coefficient", y="feature", palette="coolwarm")
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value\n(Higher => Greater Positive Impact on output=1)")
plt.ylabel("Feature")
plt.axvline(0, color='black', linestyle='--')
plt.show()

display(lr_coef_df)

# ## 9. Hyperparameter Tuning
# GridSearchCV Random Forest for optimal hyperparameters.
# search for `n_estimators` and `max_depth`.

param_grid = {
    "rf__n_estimators": [50, 100, 150],
    "rf__max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Params: {grid_search.best_params_}")
print(f"Best Score:  {grid_search.best_score_:.4f}")

best_rf_model = grid_search.best_estimator_

# Evaluate on the test set
test_preds = best_rf_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"Random Forest (GridSearchCV) Test Accuracy: {test_acc:.4f}")

# - fine-tune for even better performance.
# - more parameters (e.g., `min_samples_split`, `min_samples_leaf`, etc.) / try RandomizedSearchCV for bigger search space.