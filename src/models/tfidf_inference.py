"""
tfidf_inference.py
------------------
Load and inference logic for the TF-IDF + LR / SVM models.

Used by:
    - notebooks/05_demo.ipynb
    - flask_app/routes.py
"""

import pickle
import numpy as np


def load_model(pkl_path: str):
    """
    Loads the TF-IDF vectorizer and both classifiers from a .pkl file.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return (
        data["vectorizer"],
        data["lr"],
        data["svm"],
        data["x_test"],
        data["y_test"],
        data["label_encoder"],
    )


def predict(vectorizer, classifier, texts) -> np.ndarray:
    """
    Runs TF-IDF inference with either the LR or SVM classifier.
    """
    x_vec = vectorizer.transform(texts)
    return classifier.predict(x_vec)


def predict_single(vectorizer, classifier, label_encoder, text: str) -> str:
    """
    Convenience wrapper that predicts the specialty for a single transcription.
    """
    preds = predict(vectorizer, classifier, [text])
    return label_encoder.inverse_transform(preds)[0]
