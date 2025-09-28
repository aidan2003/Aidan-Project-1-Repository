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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show correlation of each feature with the target (Step)
print("\n\nCorrelation of features with Step:")
for col in ["X", "Y", "Z"]:
    print(f"{col} vs Step: {corr.loc['Step', col]:.3f}")
