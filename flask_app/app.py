"""
app.py
------
Flask entry point.

Loads TF-IDF + CNN-LSTM at startup (~10s on CPU).
BERT predictions served from pre-computed bert_demo_samples.json.

Run from project root:
    python flask_app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from flask_app.routes import register_routes
from src.models import tfidf_inference, cnn_lstm_inference

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("  Loading models...")

tfidf_data = tfidf_inference.load_model(
    os.path.join(BASE, 'models', 'tfidf_augmented_15.pkl')
)

cnn_lstm_data = cnn_lstm_inference.load_model(
    os.path.join(BASE, 'models', 'CNN_LSTM_augmented_mtsample.pkl')
)

BERT_DEMO = os.path.join(BASE, 'flask_app', 'static', 'bert_demo_samples.json')

print("  Ready. Starting server...")

register_routes(app, tfidf_data, cnn_lstm_data, BERT_DEMO)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)