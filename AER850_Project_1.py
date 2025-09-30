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

# Part A: Count how many times each Step appears
step_counts = df["Step"].value_counts().sort_index()

plt.bar(step_counts.index.astype(str), step_counts.values)
plt.title("Number of Samples per Step")
plt.xlabel("Step")
plt.ylabel("Frequency")
plt.show()

# Part B: Average X, Y, Z values per Step
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

# ================================
# Step 5: Model Performance Analysis
# ================================

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("\n\n================ Step 5: Model Performance Analysis ================\n")

# Logistic Regression (use best estimator from GridSearchCV)
print("\n================ Logistic Regression Results ================\n")
best_logreg = grid_logreg.best_estimator_
y_pred_test1 = best_logreg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test1, zero_division = 0))
cm1 = confusion_matrix(y_test, y_pred_test1)
sns.heatmap(cm1, annot = True, fmt = "d")
plt.title("Confusion Matrix - Logistic Regression (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Decision Tree (use best estimator from GridSearchCV)
print("\n================ Decision Tree Results ================\n")
best_tree = grid_tree.best_estimator_
y_pred_test2 = best_tree.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test2))
cm2 = confusion_matrix(y_test, y_pred_test2)
sns.heatmap(cm2, annot = True, fmt = "d")
plt.title("Confusion Matrix - Decision Tree (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()

# Random Forest (use best estimator from GridSearchCV)
print("\n================ Random Forest Results ================\n")
best_forest = grid_forest.best_estimator_
y_pred_test3 = best_forest.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_test3))
cm3 = confusion_matrix(y_test, y_pred_test3)
sns.heatmap(cm3, annot = True, fmt = "d")
plt.title("Confusion Matrix - Random Forest (Best Params)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()
