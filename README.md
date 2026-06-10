# MLOps Group Project — End-to-End Pipeline

**Group Members:** *(fill in your names and roll numbers)*

| Name | Roll Number | Contribution |
|------|-------------|--------------|
| Nikhil Saini | G25AIT2067 | Repository setup, CI/CD, Docker |
| *(Member 2)* | | |
| *(Member 3)* | | |

---

## Project Overview

This project builds a complete, production-style MLOps pipeline:
- **Data:** Choose a publicly available dataset, clean and normalize it
- **Model:** Fine-tune a compact pre-trained model from Hugging Face
- **Tracking:** Log multiple experiment versions on Weights & Biases
- **Containerisation:** Package inference with Docker
- **Automation:** Run CI and inference via GitHub Actions

---

## Submission Links (Required)

| Resource | Public Link |
|----------|-------------|
| **GitHub Repository** | https://github.com/nikhilsaini-iitj/MLOps_GroupProject |
| **Kaggle Notebook v1** | *(add after running)* |
| **Kaggle Notebook v2** | *(add after running)* |
| **Hugging Face Model** | *(add after pushing)* |
| **Docker Image** | *(add after pushing to Docker Hub / GHCR)* |
| **W&B Dashboard** | *(add after training)* |

---

## Repository Structure

```
.
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # CI: lint on push to develop
│   │   └── inference.yml       # Manual inference trigger
├── src/
│   ├── data.py               # Data loading, cleaning, encoding
│   ├── train.py              # Training script (local use)
│   ├── eval.py               # Evaluation & artifact logging
│   ├── utils.py              # Helpers, label maps, compute_metrics
│   └── inference.py          # Inference entrypoint for Docker & Actions
├── Dockerfile                # Inference container
├── requirements.txt          # Python dependencies
├── id2label.json             # Label mapping (committed)
├── README.md                 # This file
└── .gitignore
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
# Build
docker build --build-arg HF_MODEL_NAME=your-username/your-model \
  -t mlops-groupproject-inference:latest .

# Test
docker run --rm -e HF_TOKEN=$HF_TOKEN -e INPUT_TEXT="Sample text" \
  mlops-groupproject-inference:latest

# Push
docker push your-dockerhub/mlops-groupproject-inference:latest
```

### GitHub Actions

- **CI:** Automatically runs on every push to `develop`
- **Inference:** Trigger manually via "Actions → Inference → Run workflow"

---

## Dataset & Task

*(Fill in after choosing your dataset and task)*

| Attribute | Value |
|-----------|-------|
| **Task** | *(e.g., Text Classification / Image Classification)* |
| **Dataset** | *(name and source URL)* |
| **Samples** | *(total count)* |
| **Classes** | *(number of output labels)* |
| **Preprocessing** | *(cleaning steps applied)* |

---

## Model

*(Fill in after selecting your Hugging Face model)*

| Attribute | Value |
|-----------|-------|
| **Model** | *(e.g., distilbert-base-uncased)* |
| **Size** | *(parameter count / file size)* |
| **Rationale** | *(why this model suits the task)* |

---

## Experiment Results

| Version | Epochs | Batch Size | LR | Accuracy | F1 | Loss |
|---------|--------|------------|-----|----------|-----|------|
| v1 | | | | | | |
| v2 | | | | | | |

---

## License

MIT
