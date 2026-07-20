"""
train_model.py
----------------
Iris Flower Classification - Training Pipeline

Steps:
1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Preprocessing (scaling, train-test split)
4. Train multiple classifiers (KNN, Decision Tree, Logistic Regression,
   Random Forest, SVM)
5. Evaluate and compare models
6. Save the best model + scaler using Joblib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so this also runs headless (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ----------------------------------------------------------------------
# Setup paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------------------
# The Iris dataset (Fisher, 1936) has 150 samples, 4 features
# (sepal length, sepal width, petal length, petal width in cm) and
# 3 balanced classes: Setosa, Versicolor, Virginica.
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

df.to_csv(os.path.join(DATA_DIR, "iris.csv"), index=False)
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Class balance ---")
print(df["species_name"].value_counts())

# Pairplot: relationships between all feature pairs, colored by species
pairplot = sns.pairplot(df, hue="species_name", vars=iris.feature_names, diag_kind="hist")
pairplot.fig.suptitle("Pairwise Feature Relationships by Species", y=1.02)
pairplot.savefig(os.path.join(VISUALS_DIR, "pairplot.png"))
plt.close("all")

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "correlation_heatmap.png"))
plt.close()

# Boxplots per feature by species (spot outliers + spread)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.flatten(), iris.feature_names):
    sns.boxplot(data=df, x="species_name", y=feature, ax=ax, hue="species_name", legend=False)
    ax.set_title(feature)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "boxplots.png"))
plt.close()

# Class distribution bar chart
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="species_name", hue="species_name", legend=False)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "class_distribution.png"))
plt.close()

print(f"\nEDA plots saved to: {VISUALS_DIR}")

# ----------------------------------------------------------------------
# 3. PREPROCESSING: FEATURE SCALING + TRAIN-TEST SPLIT
# ----------------------------------------------------------------------
X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 4. TRAIN & COMPARE MULTIPLE MODELS
# ----------------------------------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
best_model_name = None
best_model_obj = None
best_f1 = -1

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix plot for each model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(VISUALS_DIR, f"confusion_matrix_{fname}.png"))
    plt.close()

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

# ----------------------------------------------------------------------
# 5. RESULTS COMPARISON TABLE
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
print("\n\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(BASE_DIR, "model_comparison_results.csv"), index=False)

# Bar chart comparing models
plt.figure(figsize=(8, 5))
melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison")
plt.ylim(0.8, 1.02)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "model_comparison.png"))
plt.close()

print(f"\nBest model: {best_model_name} (F1-Score: {best_f1:.4f})")

# ----------------------------------------------------------------------
# 6. SAVE BEST MODEL + SCALER
# ----------------------------------------------------------------------
joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

metadata = {
    "best_model_name": best_model_name,
    "f1_score": round(best_f1, 4),
    "feature_names": iris.feature_names,
    "target_names": list(iris.target_names),
}
with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved best model ('{best_model_name}') and scaler to: {MODELS_DIR}")
print("Training pipeline complete.")
