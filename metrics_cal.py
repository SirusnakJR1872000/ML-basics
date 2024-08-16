# Calculating Accuracy, Precision, Recall, and F1-Score from Scratch:

import numpy as np 

def calculate_metrics(y_true, y_pred):
    true_positives = sum((y_true == 1) & (y_pred == 1))
    true_negatives = sum((y_true == 0) & (y_pred == 0))
    false_positives = sum((y_true == 0) & (y_pred == 1))
    false_negatives = sum((y_true == 1) & (y_pred == 0))

    accuracy = (true_positives + true_negatives) / len(y_true)
    precision = (true_positives) / (true_positives + false_positives)
    recall = (true_positives) / (true_positives + false_negatives)
    f1_score = (precision * recall) / (recall + precision)

    return accuracy, precision, recall, f1_score

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

accuracy, precision, recall, f1_score = calculate_metrics(y_true, y_pred)



