# Step 1: Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, f1_score, recall_score
)

from math import sqrt

# Step 2: Load and Describe Dataset
heart_data = pd.read_csv("Data/heart.csv")

print("First 5 rows of the dataset:")
print(heart_data.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(heart_data.info())

# Describe the dataset statistics
print("\nDataset Description:")
print(heart_data.describe())

# Step 4: Exploratory Data Analysis (EDA)
## 4.1 Univariate Analysis (Single-variable plots)
# Age Distribution with Mean and Median
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
mean_age = heart_data['age'].mean()
median_age = heart_data['age'].median()
sns.histplot(heart_data['age'], bins=30, kde=True, color='skyblue')
plt.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.2f}')
plt.axvline(median_age, color='green', linestyle='-', label=f'Median: {median_age:.2f}')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# Maximum Heart Rate Achieved
# plt.figure(figsize=(10, 6))
# mean_thalach = heart_data['thalach'].mean()
# sns.histplot(heart_data['thalach'], bins=30, kde=True, color='lightcoral')
# plt.axvline(mean_thalach, color='blue', linestyle='--', label=f'Mean: {mean_thalach:.2f}')
# plt.title("Maximum Heart Rate Achieved")
# plt.xlabel("Maximum Heart Rate")
# plt.ylabel("Count")
# plt.legend()
# plt.show()

# Chest Pain Type Distribution with Proportions
plt.figure(figsize=(8, 5))
cp_counts = heart_data['cp'].value_counts(normalize=True) * 100
sns.countplot(x='cp', data=heart_data, palette='muted')
for i, value in enumerate(cp_counts):
    plt.text(i, cp_counts[i] / 100 * len(heart_data) + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
plt.title("Distribution of Chest Pain Types (With Proportions)")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# Target Variable Distribution (Heart Attack Risk)
plt.figure(figsize=(6, 4))
sns.countplot(x='output', data=heart_data, palette='Set2')
plt.title("Heart Attack Risk Distribution")
plt.xlabel("Output (0 = No Risk, 1 = High Risk)")
plt.ylabel("Count")
plt.show()

## 4.2 Bivariate Analysis (Two-variable plots)
# Age vs. Maximum Heart Rate with Regression Line and Correlation Coefficient
plt.figure(figsize=(10, 6))
sns.regplot(x='age', y='thalach', data=heart_data, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
correlation = heart_data['age'].corr(heart_data['thalach'])
plt.title(f"Age vs. Maximum Heart Rate (Correlation: {correlation:.2f})")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

# Chest Pain Type vs. Heart Attack Risk
plt.figure(figsize=(8, 5))
sns.countplot(x='cp', hue='output', data=heart_data, palette='viridis')
plt.title("Chest Pain Type vs. Heart Attack Risk")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.legend(title="Output")
plt.show()

# Resting Blood Pressure vs. Cholesterol (Colored by Output)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trtbps', y='chol', hue='output', data=heart_data, palette='plasma', alpha=0.8)
sns.regplot(x='trtbps', y='chol', data=heart_data, scatter=False, line_kws={'color': 'red'})
plt.title("Resting Blood Pressure vs. Cholesterol (Colored by Output)")
plt.xlabel("Resting Blood Pressure (trtbps)")
plt.ylabel("Cholesterol (chol)")
plt.legend(title="Output")
plt.show()

# Slope of Peak ST Segment vs. Heart Attack Risk
# plt.figure(figsize=(8, 5))
# sns.countplot(x='slp', hue='output', data=heart_data, palette='cubehelix')
# plt.title("Slope of Peak ST Segment vs. Heart Attack Risk")
# plt.xlabel("Slope of Peak ST Segment")
# plt.ylabel("Count")
# plt.legend(title="Output")
# plt.show()

# Step 5: Data Preparation
# Separate features (X) and target (y)
X = heart_data.drop('output', axis=1)
y = heart_data['output']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")


# Step 7: Model Performance Evaluation
def evaluate_model(model_name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1)
    print("Recall:", recall)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
evaluate_model("Decision Tree", y_test, dt_preds)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
evaluate_model("Random Forest", y_test, rf_preds)

# Logistic Regression
lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
evaluate_model("Logistic Regression", y_test, lr_preds)


# Step 8: Cross-validation
def cross_validate_model(model, X, y, model_name):
    """Perform cross-validation and print results."""
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{model_name} Cross-Validation Results:")
    print("Accuracy scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

cross_validate_model(dt_model, X, y, "Decision Tree")
cross_validate_model(rf_model, X, y, "Random Forest")
cross_validate_model(lr_model, X, y, "Logistic Regression")