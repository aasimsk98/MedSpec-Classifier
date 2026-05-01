"""
bert_inference.py
-----------------
Load and inference logic for the fine-tuned BioM-BERT-Large model.

Used by:
    - notebooks/05_demo.ipynb
    - flask_app/routes.py
"""

import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from src.preprocessing import smart_truncate_words

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pkl_path: str):
    """
    Loads the BioM-BERT-Large model from a .pkl file.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    config = AutoConfig.from_pretrained(
        data["model_name"], num_labels=data["num_classes"]
    )
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(data["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(data["model_name"])

    return (
        model,
        tokenizer,
        data["label_encoder"],
        data["max_length"],
        data["word_limit_front"],
        data["word_limit_back"],
    )


def predict(model, tokenizer, texts: list, max_length: int,
            front: int, back: int, batch_size: int = 16) -> np.ndarray:
    """
    Runs BioM-BERT inference on a list of raw transcription strings.
    """
    all_preds = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        truncated = [smart_truncate_words(t, front=front, back=back) for t in batch]

        encoding = tokenizer(
            truncated,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def predict_single(model, tokenizer, label_encoder,
                   text: str, max_length: int,
                   front: int, back: int) -> str:
    """
    Convenience wrapper that predicts the specialty for a single transcription.
    """
    preds = predict(model, tokenizer, [text], max_length, front, back)
    return label_encoder.inverse_transform(preds)[0]
