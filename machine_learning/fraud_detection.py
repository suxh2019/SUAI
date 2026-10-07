# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

'''
Credit Card Fraud Detection | Machine Learning 
• Developed a machine learning fraud detection system using Python, Pandas, and
scikit-learn to classify fraudulent financial transactions and identify suspicious
activity patterns.
• Built end-to-end ML pipelines including data preprocessing, feature scaling, model
training, and evaluation using Logistic Regression and Random Forest classifiers.
• Evaluated model performance using precision, recall, F1-score, and confusion matrix
analysis on imbalanced transaction datasets to improve fraud detection reliability.
'''
file_path = os.path.join(os.getcwd(), "machine_learning\creditcard.csv")

print(file_path)
# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(file_path)

print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

# -----------------------------
# Check Fraud Distribution
# -----------------------------
fraud_count = data["Class"].value_counts()

print("\nClass Distribution:")
print(fraud_count)

# 0 = Normal
# 1 = Fraud

# -----------------------------
# Features and Labels
# -----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# Scale Amount and Time
# -----------------------------
scaler = StandardScaler()

X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------
# Logistic Regression Model
# -----------------------------
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

print("\n===== Logistic Regression =====")

print("Accuracy:",
      accuracy_score(y_test, log_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_predictions))

print("\nClassification Report:")
print(classification_report(y_test, log_predictions))

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n===== Random Forest =====")

print("Accuracy:",
      accuracy_score(y_test, rf_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# -----------------------------
# Feature Importance
# -----------------------------
importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features:")
print(importance_df.head(10))

# -----------------------------
# Plot Top Features
# -----------------------------
top_features = importance_df.head(10)

plt.figure(figsize=(10, 5))

plt.bar(top_features["Feature"],
        top_features["Importance"])

plt.xticks(rotation=45)

plt.title("Top Important Features")

plt.tight_layout()

plt.show()