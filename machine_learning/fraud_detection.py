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
• builds a credit card fraud detection system using two supervised machine learning algorithms 
   (1) Logistic Regression — predicts whether a transaction is fraudulent using a linear classification model.
   (2) Random Forest — combines multiple decision trees to classify transactions.
It then evaluates both models using accuracy, confusion matrices, and classification reports. 
Finally, it identifies the most important features used by Random Forest.
'''
file_path = os.path.join(os.getcwd(), "machine_learning","creditcard.csv")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(file_path)
# data.shape returns:(number_of_rows, number_of_columns)
print("Dataset Shape:")
print(data.shape)

# Return the first five rows.
print("\nFirst 5 Rows:")
print(data.head())

# -----------------------------
# Check Fraud Distribution
# -----------------------------
# data["Class"]: select the target column. .value_counts()Counts how many times each unique value appears.
fraud_count = data["Class"].value_counts()

print("\nClass Distribution:")
print(fraud_count) # accuracy alone is not a good metric for fraud detection.

# 0 = Normal
# 1 = Fraud

# -----------------------------
# Features and Labels
# -----------------------------
# X = data.drop("Class", axis=1); Remove the Class column from the DataFrame. axis=1 means operate on columns.
X = data.drop("Class", axis=1) # X contains all the input features
y = data["Class"] # Extracts the target labels.

# -----------------------------
# Scale Amount and Time
# -----------------------------
# Creates an instance of scikit-learn's StandardScaler
# After scaling, the feature has approximately zero mean and unit variance on the data used to fit the scaler.
scaler = StandardScaler()
# X["Amount"]: selects the Amount column as a pandas Series. X["Amount"].shape(284807,)
#  X[["Amount"]]: selects the column as a two-dimensional DataFrame. StandardScaler expects a 2D input. X[["Amount"]].shape (284807, 1)
# Performs two operations:
#   1. fit() computes the mean and standard deviation.
#   2. transform() applies the standardization formula.
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
# create a Logistic Regression classifier. sets the maximum number of iterations used by the optimization solver.
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
# Random Forest Model: Each decision tree makes decisions by splitting the data according to feature values.
# -----------------------------
# n_estimators=100 means the model builds 100 decision trees.
# random_state=42 makes random aspects of training reproducible.
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Random Forest trains multiple trees using randomized samples of the training data and randomized feature selection.
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n===== Random Forest =====")

print("Accuracy:",
      accuracy_score(y_test, rf_predictions))
# A confusion matrix shows correct predictions and classification errors.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# -----------------------------
# Feature Importance
# -----------------------------
#rf_model.feature_importances_   extracts feature importance scores calculated by the trained Random Forest.
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