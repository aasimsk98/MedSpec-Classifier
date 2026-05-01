"""
routes.py
---------
Endpoints:
    GET  /                   → dashboard
    GET  /api/random-sample  → picks random sample, runs TF-IDF + CNN live,
                               looks up BERT prediction from pre-computed JSON
    GET  /api/health         → health check
"""

import re
import json
import random
import os
import numpy as np
import pandas as pd
from flask import render_template, jsonify

# Preprocessing for TF-IDF demo inference
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    _lem   = WordNetLemmatizer()
    _stops = set(stopwords.words('english')) - {
        'no','not','nor','against','few','more','most','other',
        'up','down','out','off','over','under','again','further'
    }
    NLTK_OK = True
except Exception:
    NLTK_OK = False


def _preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    if NLTK_OK:
        tokens = [_lem.lemmatize(t) for t in text.split()
                  if t not in _stops and len(t) > 2]
    else:
        tokens = [t for t in text.split() if len(t) > 2]
    return ' '.join(tokens)


def _label(le, idx: int) -> str:
    if isinstance(le, list):
        return le[idx]
    return le.classes_[idx]


# Route registration

def register_routes(app, tfidf_data, cnn_lstm_data, bert_demo_path: str):
    """
    bert_demo_path: path to bert_demo_samples.json
    """

    vectorizer, lr, svm, _, _, tfidf_le = tfidf_data
    cnn_model, vocab, _, _, cnn_le       = cnn_lstm_data

    # Load pre-computed BERT samples
    print(f"  Loading BERT demo samples from {bert_demo_path}...")
    with open(bert_demo_path, "r") as f:
        bert_samples = json.load(f)
    print(f"  Loaded {len(bert_samples)} pre-computed BERT samples.")

    # Dashboard
    @app.route('/')
    def index():
        return render_template('index.html')

    # Health
    @app.route('/api/health')
    def health():
        return jsonify({
            'status':         'ok',
            'bert_samples':   len(bert_samples),
            'models_loaded':  True
        })

    # Random sample 
    @app.route('/api/random-sample')
    def random_sample():
        """
        1. Pick random entry from bert_demo_samples.json
        2. Run TF-IDF (LR + SVM) + CNN-LSTM live on the transcription
        3. Return BERT pred from pre-computed JSON
        """
        try:
            sample = random.choice(bert_samples)
            text   = sample["transcription"]
            truth  = sample["ground_truth"]

            # TF-IDF live
            proc          = _preprocess(text)
            x_vec         = vectorizer.transform([proc])
            tfidf_lr_idx  = int(lr.predict(x_vec)[0])
            tfidf_svm_idx = int(svm.predict(x_vec)[0])
            tfidf_lr_pred  = _label(tfidf_le, tfidf_lr_idx)
            tfidf_svm_pred = _label(tfidf_le, tfidf_svm_idx)

            # CNN-LSTM live
            from src.models.cnn_lstm_inference import predict_single as cnn_predict
            cnn_pred = cnn_predict(cnn_model, vocab, cnn_le, text)

            # BERT from JSON
            bert_pred = sample["bert_pred"]

            return jsonify({
                'success':               True,
                'ground_truth':          truth,
                'transcription_preview': sample["transcription_preview"],
                'transcription_full':    text,
                'word_count':            sample["word_count"],
                'predictions': {
                    'tfidf_lr':  {'pred': tfidf_lr_pred,  'correct': tfidf_lr_pred  == truth},
                    'tfidf_svm': {'pred': tfidf_svm_pred, 'correct': tfidf_svm_pred == truth},
                    'cnn_lstm':  {'pred': cnn_pred,        'correct': cnn_pred       == truth},
                    'bert':      {'pred': bert_pred,        'correct': bert_pred      == truth},
                }
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500