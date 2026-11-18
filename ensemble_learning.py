
# Practical Assignment 5
# Ensemble Learning and Performance Evaluation

# 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)


# 2. Generate Synthetic Classification Dataset

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42
)

print("Dataset Generated Successfully")
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])


# 3. Split Dataset into Training and Testing Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 4. Define Base Classifiers

svm = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )
)

dt = DecisionTreeClassifier(
    random_state=42
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

gnb = GaussianNB()

base_models = {
    "SVM": svm,
    "Decision Tree": dt,
    "KNN": knn,
    "Gaussian NB": gnb
}


# 5. Implement Stacking Classifier

stacking = StackingClassifier(
    estimators=[
        ('svm', svm),
        ('dt', dt),
        ('knn', knn),
        ('gnb', gnb)
    ],
    final_estimator=LogisticRegression(
        max_iter=1000
    ),
    cv=5
)


# 6. Implement AdaBoost Classifier

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)


# 7. Implement Voting Classifier

voting = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('dt', dt),
        ('knn', knn),
        ('gnb', gnb)
    ],
    voting='hard'
)


# 8. Combine All Models

all_models = {
    **base_models,
    "Stacking": stacking,
    "AdaBoost": adaboost,
    "Hard Voting": voting
}


# 9. Train Models and Evaluate Performance

results = []

for name, model in all_models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        loss = log_loss(y_test, y_prob)
    else:
        loss = np.nan

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        loss
    ])


# 10. Create Performance Comparison Table

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Log Loss"
    ]
)

print("\nENSEMBLE LEARNING PERFORMANCE COMPARISON")

print(results_df.round(4).to_string(index=False))


# 11. Save Results as CSV File

results_df.to_csv(
    "results.csv",
    index=False
)

print("\nResults saved as results.csv")


# 12. Plot and Save Accuracy Comparison

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.title("Accuracy Comparison of ML and Ensemble Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=30)
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 13. Plot and Save F1-Score Comparison

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["F1-Score"] * 100
)

plt.title("F1-Score Comparison of ML and Ensemble Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("F1-Score (%)")

plt.xticks(rotation=30)
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "f1_score_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 14. Display Best Performing Model

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBEST PERFORMING MODEL")

print("Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")

print("\nPractical Assignment 5 Completed Successfully!")
