"""
Penguin Species Classification using Supervised Machine Learning
Author: <Your Name>
Module: Data Analysis and Machine Learning
"""

# -----------------------------
# Import required libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset (relative path)
# -----------------------------
# Make sure penguins.csv is in the "data" folder
data = pd.read_csv("data/penguins.csv")

# -----------------------------
# Data cleaning
# -----------------------------
# Drop rows with missing values
data = data.dropna()

# Select features for classification
X = data[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = data["species"]

# -----------------------------
# Exploratory data analysis
# -----------------------------
print("Dataset summary:")
print(data.describe())

# Visualize pairwise relationships between features
sns.pairplot(data, hue="species")
plt.show()

# -----------------------------
# Split data into training and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -----------------------------
# Decision Tree Classifier
# -----------------------------
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
