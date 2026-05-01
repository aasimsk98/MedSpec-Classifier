"""
cnn_lstm_inference.py
---------------------
CNNLSTMModel architecture, load and inference logic.

Used by:
    - notebooks/05_demo.ipynb
    - flask_app/routes.py
"""

import torch
import torch.nn as nn
import numpy as np

# Model Architecture

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM text classifier.

    Architecture:
        Layer 1 : Embedding (Word2Vec initialised)
        Layer 2 : TextCNN  — parallel Conv1d with kernel sizes [2, 3, 4]
                             + global max pooling
        Layer 3 : LSTM     — processes concatenated CNN output
        Layer 4 : Dropout + Linear classification head
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 200,
                 num_filters: int = 128,
                 kernel_sizes: list = None,
                 lstm_hidden: int = 200,
                 num_classes: int = 15,
                 dropout: float = 0.5,
                 pretrained_embeddings: np.ndarray = None):

        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters,
                          kernel_size=k, padding=k // 2),
                nn.ReLU()
            ) for k in kernel_sizes
        ])

        cnn_out_dim = num_filters * len(kernel_sizes)

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        emb_t = emb.permute(0, 2, 1)      # (batch, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = conv(emb_t)                        # (batch, num_filters, seq_len')
            p = torch.max(c, dim=2).values         # (batch, num_filters)
            pooled.append(p)

        cnn_out = torch.cat(pooled, dim=1)         # (batch, cnn_out_dim)
        lstm_in = cnn_out.unsqueeze(1)             # (batch, 1, cnn_out_dim)
        lstm_out, _ = self.lstm(lstm_in)           # (batch, 1, lstm_hidden)
        lstm_feat = lstm_out[:, -1, :]             # (batch, lstm_hidden)

        out = self.dropout(lstm_feat)
        return self.fc(out)                        # (batch, num_classes)

# Load

def load_model(pkl_path: str):
    import sys
    import __main__
    __main__.CNNLSTMModel = CNNLSTMModel

    data = torch.load(pkl_path, map_location="cpu", weights_only=False)
    model = data["model"]
    model.eval()
    return model, data["vocab"], data["x_test"], data["y_test"], data["label_encoder"]
    
# Inference helpers

def _encode_tokens(tokens: list, vocab: dict, max_len: int = 100) -> list:
    ids = [vocab.get(w, 1) for w in tokens]   # 1 = <UNK>
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))    # 0 = <PAD>


def predict(model, vocab: dict, x_test_tokens: list) -> np.ndarray:
    """
    Runs CNN-LSTM inference on pre-tokenised sequences.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for tokens in x_test_tokens:
            ids = _encode_tokens(tokens, vocab)
            x = torch.tensor([ids], dtype=torch.long)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).item()
            all_preds.append(pred)

    return np.array(all_preds)


def predict_single(model, vocab: dict, label_encoder: list, text: str) -> str:
    """
    Convenience wrapper that predicts the specialty for a single raw string.
    """
    tokens = text.lower().split()
    preds = predict(model, vocab, [tokens])
    return label_encoder[preds[0]]