
# Ensemble Learning and Performance Evaluation

## 1. Project Overview

This project implements different Ensemble Learning techniques using Python and Scikit-learn.

The main aim of this practical is to combine multiple machine learning classifiers and compare their performance with individual classifiers using different classification evaluation metrics.

The following ensemble techniques are implemented:

- Stacking Classifier
- AdaBoost Classifier
- Voting Classifier

## 2. Objectives

1. To understand the concept of Ensemble Learning.
2. To implement Stacking, AdaBoost and Voting techniques.
3. To combine multiple base classifiers for classification.
4. To evaluate model performance using classification metrics.
5. To compare individual and ensemble machine learning models.

## 3. Algorithms Used

### Base Classifiers

1. Support Vector Machine (SVM)
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes

### Ensemble Classifiers

1. **Stacking:** Combines multiple base classifiers using Logistic Regression as a meta-classifier.
2. **AdaBoost:** Uses Decision Tree with maximum depth 1 as a weak learner and combines multiple learners sequentially.
3. **Voting:** Combines predictions of multiple classifiers using majority voting (Hard Voting).

## 4. Dataset Description

A synthetic classification dataset is generated using Scikit-learn.

- Total Samples: 1000
- Total Features: 10
- Informative Features: 8
- Redundant Features: 2
- Training Data: 80%
- Testing Data: 20%
- Random State: 42

## 5. Technologies Used

- Programming Language: Python
- Platform: Google Colab / Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 6. Performance Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Log Loss

## 7. Project Files

| File Name | Description |
|---|---|
| ensemble_learning.py | Complete Python implementation |
| README.md | Project description and instructions |
| requirements.txt | Required Python libraries |
| results.csv | Performance comparison results |
| accuracy_comparison.png | Accuracy comparison graph |
| f1_score_comparison.png | F1-score comparison graph |
| Ensemble_Learning.ipynb | Google Colab notebook (optional) |

## 8. Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## 9. How to Run the Project

1. Download or clone this repository.
2. Open the project folder.
3. Install the required Python libraries.
4. Run the Python file using:

```bash
python ensemble_learning.py
```

5. The program generates the performance comparison table.
6. The accuracy and F1-score graphs are displayed.
7. The results are saved in CSV format.

## 10. Expected Output

The program displays:

- Dataset information
- Training and testing sample counts
- Performance comparison of all seven models
- Accuracy comparison graph
- F1-score comparison graph
- Best performing model

The program also generates:

- results.csv
- accuracy_comparison.png
- f1_score_comparison.png

Actual performance values are obtained when the program is executed.

## 11. Conclusion

In this practical, different Ensemble Learning techniques such as Stacking, AdaBoost and Voting Classifier are implemented successfully.

The performance of these techniques is compared with individual machine learning classifiers using different classification metrics.

This practical demonstrates how combining multiple classifiers can help improve classification performance and provides an understanding of the working of different ensemble learning techniques.



