# MedSpec Classifier

Automatic classification of clinical transcriptions into **15 medical specialties** : comparing TF-IDF, CNN-LSTM, and BioM-BERT-Large under identical data + evaluation conditions.

**Live Demo → [bit.ly/medspec-classifier](https://bit.ly/medspec-classifier)**

---

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Weighted F1 | Macro F1 |
|-------|----------|----------------|--------------|-------------|----------|
| TF-IDF + LR | 76.22% | 76.62% | 77.49% | 0.7650 | 0.7651 |
| TF-IDF + SVM | 75.70% | 75.46% | 76.91% | 0.7573 | 0.7587 |
| CNN-LSTM | **81.97%** | 80.98% | 82.71% | 0.8217 | 0.8120 |
| BioM-BERT-Large | 81.49% | 81.72% | 83.40% | 0.8200 | **0.8162** |

> CNN-LSTM = best accuracy. BioM-BERT = best Macro F1 → recommended for imbalanced 15-class eval.

### Key Finding
Data decisions > model architecture:

| Decision | Accuracy Gain |
|----------|--------------|
| Removed document-type label categories | +21% |
| Back-translation augmentation | +19% |
| GroupShuffleSplit over standard split | +4% |

---

## BioM-BERT — 6 Experiment Progression

| Exp | Dataset | Specialties | Split | Accuracy | Macro F1 |
|-----|---------|-------------|-------|----------|----------|
| 1 | Raw | 20 | TTS | 37.7% | 0.413 |
| 2 | Raw | 20 | GSS | 42.4% | 0.445 |
| 3 | Augmented | 20 | TTS | 53.6% | 0.549 |
| 4 | Augmented | 20 | GSS | 58.9% | 0.584 |
| 5 | Augmented | 15 | TTS | 78.1% | 0.783 |
| **6** | **Augmented** | **15** | **GSS** | **81.5%** | **0.816** |

All 6 use identical BioM-BERT-Large architecture — only data config + split vary.

---

## Dataset

**MTSamples** — 4,999 clinical transcriptions, 40 specialties ([Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions))

Preprocessing pipeline:
1. Removed specialties < 61 samples
2. Back-translation augmentation (EN → FR → EN) on minority classes
3. Removed 5 document-type categories (Surgery, Consult - History and Phy., Radiology, SOAP/Chart/Progress Notes, Discharge Summary)
4. Final: **5,436 transcriptions × 15 genuine medical specialties**

---

## Project Structure

```
MedSpec-Classifier/
│
├── ├── notebooks/
│   ├── 01_data_cleaning_augmentation.ipynb
│   ├── 02_tfidf.ipynb
│   ├── 03_lstm.ipynb
│   ├── 04_biom_bert.ipynb
│   ├── 05_demo.ipynb
│   ├── BioM_BERT_Demo_Samples.ipynb
│   └── BioM_BERT/
│       ├── 04_biom_bert.ipynb
│       ├── all_experiments_results.csv
│       ├── results_comparison.png
│       ├── label_encoder_aug15.pkl
│       ├── label_encoder_aug20.pkl
│       ├── label_encoder_raw20.pkl
│       ├── class_distributions/
│       ├── confusion_matrices/
│       ├── training_histories/
│       ├── shared_tokenizer/
│       └── models_pkl/              # Drive only — see models_pkl/README.md
│
├── src/
│   ├── preprocessing.py                 # smart_truncate_words
│   ├── evaluate.py                      # calc_results, plot_confusion_matrix
│   └── models/
│       ├── bert_inference.py
│       ├── cnn_lstm_inference.py
│       └── tfidf_inference.py
│
├── flask_app/                           # Web dashboard
│   ├── app.py
│   ├── routes.py
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       ├── js/dashboard.js
│       └── bert_demo_samples.json       # Pre-computed BERT predictions on test set
│
├── models/                              # see models/README.md
├── data/                                # CSVs — included 
├── Dockerfile
├── requirements.txt                     # full dev dependencies
├── requirements_deploy.txt             # lean prod dependencies
└── DEPLOY.md                           # full Azure deployment guide
```

---

## Setup

```bash
git clone https://github.com/aasimsk98/MedSpec-Classifier.git
cd MedSpec-Classifier
pip install -r requirements.txt
```

### Download Models

PKL files + BERT model → **[Google Drive](https://drive.google.com/drive/folders/1IjCPsjzEalWR7UGvBGcnJ_MtI1Pr0F0A?usp=drive_link)**

Place downloaded files:
```
models/tfidf_augmented_15.pkl
models/CNN_LSTM_augmented_mtsample.pkl
models/exp6_aug15_gss.pkl              # BERT only — needed for notebooks
```

### Run Notebooks
Open in Jupyter or Google Colab. BioM-BERT notebook requires GPU (A100 recommended, ~3hrs).

### Run Web App Locally

```bash
python flask_app/app.py
```

---

## Deployment

Containerized with Docker, deployed to **Azure Container Apps** (free tier).

Full guide → [DEPLOY.md](DEPLOY.md)

```bash
docker build -t medspec-classifier .
docker push aasimsk98/medspec-classifier:latest
```

**Live:** [bit.ly/medspec-classifier](https://bit.ly/medspec-classifier)

---

## Models

### BioM-BERT-Large
- Base: `sultan/BioM-BERT-PubMed-PMC-Large` (334M params, full fine-tune)
- Input: smart truncation — first 175 + last 175 words
- Loss: CrossEntropyLoss + inverse-frequency class weights
- Optimizer: AdamW (lr=1.5e-5, weight_decay=0.01, warmup_ratio=0.1)
- Early stopping: patience=2 on val Macro F1

### CNN-LSTM
- Word2Vec skip-gram embeddings (dim=200)
- TextCNN parallel filters [2, 3, 4] + global max pooling
- LSTM hidden=200, dropout=0.5

### TF-IDF
- Custom medical tokenizer (ICD codes, dosages, vitals)
- PMI-based bigram detection
- max_features=50,000, sublinear TF scaling
- Two classifiers: Logistic Regression + LinearSVC

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `Flask` · `Docker` · `Azure Container Apps`