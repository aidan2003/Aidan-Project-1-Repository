# ================================
# Step 1: Data Processing
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("data/Project 1 Data.csv")

# Remove missing data if any
df = df.dropna().reset_index(drop = True)

# Make sure Step is integer
df["Step"] = df["Step"].astype(int)

# ================================
# Step 2: Data Visualization
# ================================

print("\n\n================ Step 2: Data Visualization ================\n\n")

# How many times each Step appears
step_counts = df["Step"].value_counts().sort_index()

plt.bar(step_counts.index.astype(str), step_counts.values)
plt.title("Number of Samples per Step")
plt.xlabel("Step")
plt.ylabel("Frequency")
plt.show()

# Average X, Y, Z values per Step
step_means = df.groupby("Step")[["X", "Y", "Z"]].mean()

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.bar(step_means.index.astype(str), step_means["X"])
plt.title("Average X per Step")
plt.xlabel("Step"); plt.ylabel("X")

plt.subplot(1, 3, 2)
plt.bar(step_means.index.astype(str), step_means["Y"])
plt.title("Average Y per Step")
plt.xlabel("Step"); plt.ylabel("Y")

plt.subplot(1, 3, 3)
plt.bar(step_means.index.astype(str), step_means["Z"])
plt.title("Average Z per Step")
plt.xlabel("Step"); plt.ylabel("Z")

plt.tight_layout()
plt.show()

step_stats = df.groupby("Step")[["X", "Y", "Z"]].agg(["mean", "std", "min", "max"])
print(step_stats.round(2))

# ================================
# Step 3: Correlation Analysis
# ================================

print("\n\n================ Step 3: Correlation Analysis ================\n\n")

corr = df[["X", "Y", "Z", "Step"]].corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Train/Test Split

from sklearn.model_selection import train_test_split

X = df[["X", "Y", "Z"]]
y = df["Step"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Show correlation of each feature with the target (Step)
print("\n\nCorrelation of features with Step:")
for col in ["X", "Y", "Z"]:
    print(f"{col} vs Step: {corr.loc['Step', col]:.3f}")

# ================================
# Step 4: Classification Model Development
# ================================

print("\n\n================ Step 4: Classification Model Development ================\n\n")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# --------------------------------
# Logistic Regression (Base Model)
# --------------------------------
mdl1 = LogisticRegression(max_iter = 1000)
mdl1.fit(X_train, y_train)

print("Logistic Regression Training Accuracy:", round(mdl1.score(X_train, y_train), 3))
print("Logistic Regression Test Accuracy:", round(mdl1.score(X_test, y_test), 3))
cv_scores1 = cross_val_score(mdl1, X_train, y_train, cv = 5, scoring = "accuracy")
print("Logistic Regression Mean CV Accuracy:", round(cv_scores1.mean(), 3))

# Logistic Regression with GridSearchCV
param_grid_logreg = {
    'C': [0.1, 1, 10],          
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2']
}
grid_logreg = GridSearchCV(
    estimator = LogisticRegression(max_iter = 1000),
    param_grid = param_grid_logreg,
    scoring = 'accuracy',
    cv = 5
)
grid_logreg.fit(X_train, y_train)

print("Best Logistic Regression Params:", grid_logreg.best_params_)
print("Best Logistic Regression CV Accuracy:", round(grid_logreg.best_score_, 3))

# --------------------------------
# Decision Tree (Base Model)
# --------------------------------
mdl2 = DecisionTreeClassifier(random_state = 42)
mdl2.fit(X_train, y_train)

print("\nDecision Tree Training Accuracy:", round(mdl2.score(X_train, y_train), 3))
print("Decision Tree Test Accuracy:", round(mdl2.score(X_test, y_test), 3))
cv_scores2 = cross_val_score(mdl2, X_train, y_train, cv = 5, scoring = "accuracy")
print("Decision Tree Mean CV Accuracy:", round(cv_scores2.mean(), 3))

# Decision Tree with GridSearchCV
param_grid_tree = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_tree = GridSearchCV(
    estimator = DecisionTreeClassifier(random_state = 42),
    param_grid = param_grid_tree,
    scoring = 'accuracy',
    cv = 5
)
grid_tree.fit(X_train, y_train)

print("Best Decision Tree Params:", grid_tree.best_params_)
print("Best Decision Tree CV Accuracy:", round(grid_tree.best_score_, 3))

# --------------------------------
# Random Forest (Base Model)
# --------------------------------
mdl3 = RandomForestClassifier(random_state = 42)
mdl3.fit(X_train, y_train)

print("\nRandom Forest Training Accuracy:", round(mdl3.score(X_train, y_train), 3))
print("Random Forest Test Accuracy:", round(mdl3.score(X_test, y_test), 3))
cv_scores3 = cross_val_score(mdl3, X_train, y_train, cv = 5, scoring = "accuracy")
print("Random Forest Mean CV Accuracy:", round(cv_scores3.mean(), 3))

# Random Forest with GridSearchCV
param_grid_forest = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
grid_forest = GridSearchCV(
    estimator = RandomForestClassifier(random_state = 42),
    param_grid = param_grid_forest,
    scoring = 'accuracy',
    cv = 5 
)
grid_forest.fit(X_train, y_train)

print("Best Random Forest Params (GridSearchCV):", grid_forest.best_params_)
print("Best Random Forest CV Accuracy:", round(grid_forest.best_score_, 3))

# Random Forest with RandomizedSearchCV
param_dist_forest = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
rand_forest = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state = 42),
    param_distributions = param_dist_forest,
    n_iter = 5,
    scoring = 'accuracy',
    cv = 5,
    random_state = 42
)
rand_forest.fit(X_train, y_train)

print("Best Random Forest Params (RandomizedSearchCV):", rand_forest.best_params_)
print("Best Random Forest CV Accuracy:", round(rand_forest.best_score_, 3))

print("\n---- Threshold Analysis (One-vs-Rest) ----")

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Positive class
POS_CLASS = 1

# Binarize
y_test_bin = (y_test == POS_CLASS).astype(int)

# Collect tuned models
best_logreg = grid_logreg.best_estimator_
best_tree   = grid_tree.best_estimator_
best_forest = grid_forest.best_estimator_

models_for_threshold = [
    ("Logistic Regression (Best)", best_logreg),
    ("Decision Tree (Best)", best_tree),
    ("Random Forest (Best)", best_forest),
]

thresholds = np.arange(0.0, 1.0, 0.1)

best_thresholds_summary = {}

# Perform threshold analysis for each tuned model
for name, model in models_for_threshold:
    class_index = list(model.classes_).index(POS_CLASS)
    y_scores = model.predict_proba(X_test)[:, class_index]

    precisions, recalls, f1s = [], [], []
    print(f"\n{name}: Precision/Recall/F1 across thresholds (positive class = {POS_CLASS})")
    for t in thresholds:
        y_pred_thr = (y_scores >= t).astype(int)
        p = precision_score(y_test_bin, y_pred_thr, zero_division=0)
        r = recall_score(y_test_bin, y_pred_thr, zero_division=0)
        f = f1_score(y_test_bin, y_pred_thr, zero_division=0)
        precisions.append(p); recalls.append(r); f1s.append(f)
        print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

    best_idx = int(np.argmax(f1s))
    best_thresholds_summary[name] = {
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "best_f1": float(f1s[best_idx])
    }

    # Combined plots
    plt.figure(figsize = (6,4))
    plt.plot(thresholds, precisions, marker = "o", label = "Precision")
    plt.plot(thresholds, recalls, marker = "o", label = "Recall")
    plt.plot(thresholds, f1s, marker = "o", label = "F1")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title(f"{name}: Precision / Recall / F1 vs Threshold")
    plt.legend(); plt.grid(True); plt.show()

# Summary
print("\nBest thresholds by model (max F1, one-vs-rest on Step =", POS_CLASS, "):")
for k, v in best_thresholds_summary.items():
    print(f"{k}: t* = {v['best_threshold']:.1f}  (P = {v['best_precision']:.3f}, "
          f"R = {v['best_recall']:.3f}, F1 = {v['best_f1']:.3f})")

# ================================
# Step 5: Model Performance Analysis
# ================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("\n\n================ Step 5: Model Performance Analysis ================\n")

# Logistic Regression Performance Analysis
print("\n================ Logistic Regression Results ================\n")
best_logreg = grid_logreg.best_estimator_
y_pred_test1 = best_logreg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test1, zero_division = 0))
cm1 = confusion_matrix(y_test, y_pred_test1)
sns.heatmap(cm1, annot = True, fmt = "d")
plt.title("Confusion Matrix - Logistic Regression (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Decision Tree Performance Analysis
print("\n================ Decision Tree Results ================\n")
best_tree = grid_tree.best_estimator_
y_pred_test2 = best_tree.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test2))
cm2 = confusion_matrix(y_test, y_pred_test2)
sns.heatmap(cm2, annot = True, fmt = "d")
plt.title("Confusion Matrix - Decision Tree (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Random Forest Performance Analysis
print("\n================ Random Forest Results ================\n")
best_forest = grid_forest.best_estimator_
y_pred_test3 = best_forest.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test3))
cm3 = confusion_matrix(y_test, y_pred_test3)
sns.heatmap(cm3, annot = True, fmt = "d")
plt.title("Confusion Matrix - Random Forest (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# ================================
# Step 6: Stacked Model Performance Analysis
# ================================

from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\n================ Step 6: Stacked Model Performance Analysis ================\n")

# Stack Logistic Regression and Decision Tree
estimators = [
    ('logreg', mdl1),
    ('dtree', mdl2)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(random_state = 42)
)
stack_model.fit(X_train, y_train)

# Evaluate the stacked model
print("\n================ Stacked Model Results ================\n")
y_pred_stack = stack_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_stack, zero_division = 0))

cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot = True, fmt = "d")
plt.title("Confusion Matrix - Stacked Model")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# ================================
# Step 7: Model Evaluation
# ================================

import joblib
import pandas as pd

print("\n================ Step 7: Model Evaluation ================\n")

# Save the tuned best model
best_forest = grid_forest.best_estimator_
joblib.dump(best_forest, "best_model.pkl")
print("\nModel saved as best_model.pkl")

# Load and use it
loaded_model = joblib.load("best_model.pkl")

# Define the test points as full (X, Y, Z) coordinates
test_points = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.3, 3.0625, 1.93],
    [9.4, 3.0, 1.8],
    [9.4, 3.0, 1.3]
], columns = ["X", "Y", "Z"])

# Make predictions
predictions = loaded_model.predict(test_points)

# Print results
print("\nPredicted Steps for Test Points:")
for point, pred in zip(test_points.values, predictions):
    formatted_point = " ".join([f"{x:.4f}" for x in point])
    print(f"Input [{formatted_point}] = Predicted Step: {pred}")
    