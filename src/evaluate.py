"""
evaluate.py
-----------
Shared evaluation utilities used across all three models.
Extracted from demo.ipynb.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calc_results(y_true, y_pred) -> dict:
    """
    Computes the five evaluation metrics used throughout this project.

    Args:
        y_true : array-like of true integer labels
        y_pred : array-like of predicted integer labels

    Returns:
        dict with keys: accuracy, macro_precision, macro_recall,
                        weighted_f1, macro_f1
    """
    return {
        "accuracy":        accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_results(results: dict) -> None:
    """Pretty-prints the results dict returned by calc_results."""
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Macro Precision:   {results['macro_precision']:.4f}")
    print(f"Macro Recall:      {results['macro_recall']:.4f}")
    print(f"Weighted F1 Score: {results['weighted_f1']:.4f}")
    print(f"Macro F1 Score:    {results['macro_f1']:.4f}")


def plot_confusion_matrix(y_true, y_pred, label_encoder, title: str) -> None:
    """
    Plots a normalised confusion matrix using seaborn.

    Handles both sklearn LabelEncoder objects (BioM-BERT, TF-IDF)
    and plain Python lists (CNN-LSTM).

    Args:
        y_true        : pd.Series of true integer labels
        y_pred        : pd.Series of predicted integer labels
        label_encoder : sklearn LabelEncoder OR list of class name strings
        title         : plot title string
    """
    numeric_labels = sorted(y_true.unique())

    # CNN-LSTM stores label_encoder as a plain list
    if isinstance(label_encoder, list):
        string_labels = [label_encoder[i] for i in numeric_labels]
    else:
        string_labels = [label_encoder.classes_[i] for i in numeric_labels]

    conf_mat = confusion_matrix(y_true, y_pred,
                                labels=numeric_labels,
                                normalize='true')

    plt.figure(figsize=(15, 10))
    sns.heatmap(conf_mat,
                annot=True,
                cmap="Blues",
                fmt=".2f",
                xticklabels=string_labels,
                yticklabels=string_labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
