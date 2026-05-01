# Deployment Guide — Azure Container Apps (Free Tier)

## Architecture

```
Local code
    → docker build
    → Docker Hub (free public registry)
    → Azure Container Apps pulls image
    → Live at: https://medspec-classifier.blackhill-1941bca7.eastus.azurecontainerapps.io
    → Shortened: https://bit.ly/medspec-classifier
```

## Why Container Apps

- Scales to zero when idle → $0 when no traffic
- Free tier: 180,000 vCPU-sec + 360,000 GB-sec/month
- No storage limit issues (unlike App Service F1)
- Cold start ~20-30s after idle period

---

## Prerequisites

- Docker Desktop installed + running
- Docker Hub account (free) — https://hub.docker.com
- Azure account — https://portal.azure.com

---

## Step 1 - Files in Place

Confirm before building:

```
MedSpec-Classifier/
├── Dockerfile
├── requirements_deploy.txt
├── .dockerignore
├── src/
├── flask_app/
│   └── static/
│       └── bert_demo_samples.json   ← generate via notebooks/BioM_BERT_Demo_Samples.ipynb
└── models/
    ├── tfidf_augmented_15.pkl
    └── CNN_LSTM_augmented_mtsample.pkl
```

> BERT model (exp6_aug15_gss.pkl) NOT included — too large.
> BERT predictions served from bert_demo_samples.json instead.

---

## Step 2 - Generate bert_demo_samples.json (One Time)

Run `notebooks/BioM_BERT_Demo_Samples.ipynb` on Google Colab (GPU runtime).

Upload to Colab:
- `exp6_aug15_gss.pkl`
- `augmented_mtsamples.csv`
- `shared_tokenizer/` folder (optional, speeds up tokenizer load)

Download output `bert_demo_samples.json` → place at `flask_app/static/bert_demo_samples.json`.

---

## Step 3 - Build + Test Locally

```bash
cd MedSpec-Classifier

# Build image
docker build -t medspec-classifier .

# Test locally
docker run -p 8000:8000 medspec-classifier

```

---

## Step 4 - Push to Docker Hub

```bash
docker login

# Replace YOUR_USERNAME
docker tag medspec-classifier YOUR_USERNAME/medspec-classifier:latest
docker push YOUR_USERNAME/medspec-classifier:latest
```
---

## Step 5 - Deploy via Azure Portal

### 5a - Resource Group
Portal → "Resource groups" → Create

### 5b - Container App
Portal → "Container Apps" → Create

Create new environment.
 
Container settings:
```
Image source > Docker Hub (Public)
Image + tag  > YOUR_USERNAME/medspec-classifier:latest
CPU          > 0.5
Memory       > 1.0 Gi
```
 
Ingress:
```
Enable ingress          > ✓
Traffic                 > Accepting from anywhere
Type                    > HTTP
Port                    > 8000
```
→ Create. Wait 3-5 min.

### 5c — Enable Scale to Zero
Container App → Scale tab
```
Min replicas > 0
Max replicas > 1
```
Save.

---

## Updating

Code change → rebuild + push → restart:

```bash
docker build -t medspec-classifier .
docker tag medspec-classifier YOUR_USERNAME/medspec-classifier:latest
docker push YOUR_USERNAME/medspec-classifier:latest
```

Portal → Container App → Containers → Refresh image
(or wait ~5 min auto-pull)

---