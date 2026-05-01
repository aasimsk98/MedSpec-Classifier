# Models

Two deployment models tracked in repo. BioM-BERT excluded — too large.

## In This Folder

| File | Size | Description |
|------|------|-------------|
| `tfidf_augmented_15.pkl` | ~16MB | TF-IDF vectorizer + LR + SVM + test data |
| `CNN_LSTM_augmented_mtsample.pkl` | ~18MB | CNN-LSTM model + vocab + test data |

## Google Drive

Full model archive (incl. BioM-BERT exp6):
**[MedSpec-Classifier/models/](https://drive.google.com/drive/folders/1IjCPsjzEalWR7UGvBGcnJ_MtI1Pr0F0A?usp=drive_link)**

| File | Size | Description |
|------|------|-------------|
| `exp6_aug15_gss.pkl` | ~1.2GB | BioM-BERT-Large fine-tuned weights (best model) |

## After Download

Place files in `models/` folder, then run:

    python flask_app/app.py