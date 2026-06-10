# MLOps Group Project — Emotion Classification Pipeline

**Group 2 · IIT Jodhpur**

| Name | Roll Number | Contribution |
|------|-------------|--------------|
| Nikhil Saini | G25AIT2067 | Repo setup, CI/CD, Docker, report |
| Y Sharathchandrika | G25AIT2132 | Data preparation, cleaning |
| Sarthak Kapoor | G25AIT2098 | Model training, W&B tracking |
| Aryaveer Rathi | G25AIT2021 | Inference, Docker container |

---

## Project Overview

End-to-end MLOps pipeline for text emotion classification using the `dair-ai/emotion` dataset (6 classes: sadness, joy, love, anger, fear, surprise). Two model versions were trained and compared on Kaggle GPU.

---

## Submission Links (All Public)

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/nikhilsaini-iitj/MLOps_GroupProject |
| **Kaggle Notebook v1 (MiniLM)** | https://www.kaggle.com/code/nikhilg25ait2067/ |
| **Kaggle Notebook v2 (ELECTRA)** | https://www.kaggle.com/code/nikhilg25ait2067/ |
| **Hugging Face Model v1** | https://huggingface.co/Nikhil-iitj/emotion-minilm |
| **Hugging Face Model v2** | https://huggingface.co/Nikhil-iitj/emotion-electra |
| **W&B Dashboard** | https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-groupproject |
| **Docker Image** | *(add after push)* |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | `dair-ai/emotion` (Hugging Face `datasets`) |
| **Classes** | 6 — sadness, joy, love, anger, fear, surprise |
| **Size** | 16,000 train / 2,000 validation / 2,000 test |
| **Preprocessing** | Lowercase, strip punctuation, collapse whitespace, drop duplicates |

---

## Experiment Results

| Version | Model | Epochs | Batch | LR | Accuracy | F1 | Loss | Size | Time |
|---------|-------|--------|-------|-----|----------|-----|------|------|------|
| **v1** | MiniLM-L12 | 3 | 16 | 3e-5 | 0.917 | 0.9169 | 0.542 | 133.5 MB | ~3m 44s |
| **v2** | ELECTRA-small | 4 | 16 | 5e-5 | **0.920** | **0.9209** | **0.440** | **54.2 MB** | ~3m |

**Winner: v2 (ELECTRA-small)** — smaller, faster, and more accurate.

---

## Repository Structure

```
.
├── .github/workflows/
│   ├── ci.yml              # Lint on push to develop
│   └── inference.yml       # Manual inference trigger
├── src/
│   ├── data.py            # Data loading + cleaning
│   ├── train.py           # Training script (local)
│   ├── eval.py            # Evaluation + artifacts
│   ├── utils.py           # Shared helpers
│   └── inference.py       # Docker / Actions entrypoint
├── kaggle_minilm.ipynb    # Kaggle notebook v1
├── kaggle_electra.ipynb   # Kaggle notebook v2
├── Dockerfile             # Inference container
├── requirements.txt
├── id2label.json          # Label mapping
└── README.md              # This file
```

---

## Setup Instructions

### Local Development

```bash
git clone https://github.com/nikhilsaini-iitj/MLOps_GroupProject.git
cd MLOps_GroupProject
pip install -r requirements.txt
```

### Docker Build & Run

```bash
docker build --build-arg HF_MODEL_NAME=Nikhil-iitj/emotion-electra \
  -t mlops-groupproject-inference:latest .

docker run --rm -e HF_TOKEN=$HF_TOKEN -e INPUT_TEXT="I am so happy today!" \
  mlops-groupproject-inference:latest
```

### GitHub Actions

- **CI:** Automatically runs on every push to `develop`
- **Inference:** Trigger manually via "Actions → Inference → Run workflow"

---

## License

MIT
