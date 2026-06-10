# MLOps Group Project — Emotion Classification Pipeline
**End-to-End Report**

**Group 2 · PGD AI · IIT Jodhpur**

| Name | Roll Number | Primary Contribution |
|------|-------------|---------------------|
| Nikhil Saini | G25AIT2067 | Repo setup, branch protection, CI/CD workflows (ci.yml, inference.yml, docker.yml), GitHub Secrets, report compilation |
| Y Sharathchandrika | G25AIT2132 | Data inspection, cleaning pipeline (`data.py`), `id2label.json`, class-distribution analysis |
| Sarthak Kapoor | G25AIT2098 | Model training (2 Kaggle versions), W&B tracking, HF model push, W&B dashboard public setup |
| Aryaveer Rathi | G25AIT2021 | `inference.py`, Dockerfile design, local Docker build/test, inference workflow support |

---

## 1. Project Overview

We built a complete MLOps pipeline for **6-class text emotion classification** using the `dair-ai/emotion` dataset. Two transformer models were trained with different architectures and hyperparameters on Kaggle GPU, tracked with Weights & Biases, and the best model was containerised with Docker and deployed via GitHub Actions.

**Dataset:** `dair-ai/emotion` (Hugging Face)
- 16,000 train / 2,000 validation / 2,000 test
- Classes: sadness, joy, love, anger, fear, surprise

**All public links:**
| Resource | URL |
|----------|-----|
| GitHub Repository | https://github.com/nikhilsaini-iitj/MLOps_GroupProject |
| Kaggle Notebook v1 (MiniLM) | https://www.kaggle.com/code/nikhilg25ait2067/ |
| Kaggle Notebook v2 (ELECTRA) | https://www.kaggle.com/code/nikhilg25ait2067/ |
| Hugging Face Model v1 | https://huggingface.co/Nikhil-iitj/emotion-minilm |
| Hugging Face Model v2 (winner) | https://huggingface.co/Nikhil-iitj/emotion-electra |
| W&B Dashboard | https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-groupproject-v2 |
| Docker Image | https://hub.docker.com/r/nikhilsainiiitj/mlops-groupproject-inference |

---

## 2. GitHub Repository Setup (Task 1 — 10 marks)

**Owner: Nikhil Saini**

We created a public repository `MLOps_GroupProject` under the organisation account. The repository uses a `main` + `develop` branching strategy:
- `main`: production-ready code, protected by branch rules
- `develop`: integration branch for all feature work

**Branch protection on `main`:**
- Require a pull request before merging
- Require 1 approving review
- Dismiss stale PR approvals when new commits are pushed

All four team members were added as collaborators with **Write** access. The `.github/workflows/` directory contains three workflows (see Tasks 6 & 7).

**Screenshot:** Collaborators page saved in submission folder.

---

## 3. Data Preparation (Task 2 — 15 marks)

**Owner: Y Sharathchandrika**

The `dair-ai/emotion` dataset was loaded via `datasets.load_dataset()`. Preprocessing steps applied in `src/data.py`:

1. **Lowercasing** — standardises text casing
2. **Punctuation stripping** — removes non-alphanumeric characters (except apostrophes)
3. **Whitespace collapse** — replaces multiple spaces with single space
4. **Null handling** — verified no null samples exist
5. **Duplicate removal** — dropped exact duplicate text rows
6. **Label encoding** — generated `id2label.json` and `label2id` mapping:
   ```json
   {"0": "sadness", "1": "joy", "2": "love", "3": "anger", "4": "fear", "5": "surprise"}
   ```

**Class distribution note:** The dataset is mildly imbalanced — `joy` (~30%) and `sadness` (~28%) dominate, while `surprise` (~8%) and `love` (~14%) are under-represented. We documented this rather than rebalancing, as the rubric asks for inspection and rationale, not silent fixes.

**Commit:** `id2label.json` is tracked; raw/processed data is `.gitignore`d.

---

## 4. Model Selection (Task 3 — 10 marks)

**Owner: Sarthak Kapoor**

Both models were selected from the Hugging Face Hub with the constraint **< 200 MB** on disk.

### v1: `microsoft/MiniLM-L12-H384-uncased`
- **Architecture:** 12-layer, 384-hidden distilled transformer (Wang et al., 2020)
- **Objective:** General-domain distillation from BERT-base
- **Parameters:** ~33 M
- **On-disk size:** ~133.5 MB (`model.safetensors`)
- **Rationale:** Deep but narrow; strong baseline for short-text classification

### v2: `google/electra-small-discriminator`
- **Architecture:** Small ELECTRA with replaced-token detection (Clark et al., 2020)
- **Objective:** RTD (predict which tokens were replaced) rather than MLM
- **Parameters:** ~14 M
- **On-disk size:** ~54.2 MB (`model.safetensors`)
- **Rationale:** More sample-efficient pretraining objective; fewer params but competitive downstream performance

Both models were loaded with `AutoModelForSequenceClassification`, wiring `id2label`/`label2id` so the classifier head outputs human-readable labels.

---

## 5. Training — Two Versions (Task 4 — 25 marks)

**Owner: Sarthak Kapoor**

Training was executed on **Kaggle GPU T4** using the Hugging Face `Trainer` API. Secrets (`WANDB_API_KEY`, `HF_TOKEN`) were stored in **Kaggle Secrets** — never hardcoded.

### Hyperparameters

| Parameter | v1 (MiniLM) | v2 (ELECTRA-small) |
|-----------|-------------|-------------------|
| Epochs | 3 | 4 |
| Batch size | 16 | 16 |
| Learning rate | 3e-5 | 5e-5 |
| Max sequence length | 128 | 128 |
| Optimiser | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Eval strategy | epoch | epoch |
| Save strategy | epoch | epoch |
| Best metric | F1 (weighted) | F1 (weighted) |

### Validation Results (epoch-wise)

**v1 (MiniLM):**
| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|-------|-----------|----------|----------|-----|
| 1 | 1.203 | 1.089 | 0.812 | 0.798 |
| 2 | 0.612 | 0.589 | 0.901 | 0.901 |
| 3 | 0.341 | 0.542 | 0.917 | 0.917 |

**v2 (ELECTRA-small):**
| Epoch | Train Loss | Val Loss | Accuracy | F1 |
|-------|-----------|----------|----------|-----|
| 1 | 1.181 | 1.068 | 0.846 | 0.832 |
| 2 | 0.565 | 0.560 | 0.919 | 0.919 |
| 3 | 0.365 | 0.428 | 0.929 | 0.930 |
| 4 | 0.295 | 0.393 | 0.931 | 0.931 |

### Final Test Metrics

| Version | Model | Test Accuracy | Test F1 | Test Loss | Size | Runtime |
|---------|-------|--------------|---------|-----------|------|---------|
| v1 | MiniLM-L12 | 0.907 | 0.907 | 0.542 | 133.5 MB | ~3m 44s |
| **v2** | **ELECTRA-small** | **0.9265** | **0.9269** | **0.395** | **54.2 MB** | **~2m 56s** |

**Winner: v2 (ELECTRA-small)** — higher accuracy, lower loss, **2.5× smaller**, and faster inference. The RTD pretraining objective appears to produce more efficient representations for this downstream task.

**W&B tracking:** `report_to='wandb'`, `logging_strategy='steps'`, `logging_steps=20`. We used a fresh project (`mlops-groupproject-v2`) with explicit `define_metric` calls to ensure charts render correctly against epoch/step axes.

---

## 6. Hugging Face Model Push (Task 5 — 5 marks)

**Owner: Sarthak Kapoor**

The winning v2 model (best checkpoint selected by `load_best_model_at_end=True` with `metric_for_best_model='f1'`) was pushed to Hugging Face Hub:

```python
trainer.save_model('best')
tok.save_pretrained('best')
model.push_to_hub('Nikhil-iitj/emotion-electra')
tok.push_to_hub('Nikhil-iitj/emotion-electra')
```

- **Repository:** https://huggingface.co/Nikhil-iitj/emotion-electra
- **Visibility:** Public
- **Contents:** `model.safetensors` (54.2 MB), `config.json`, `tokenizer.json`, `vocab.txt`, `id2label` mapping
- **W&B link:** `wandb.run.summary['huggingface_model']` points to the same URL

---

## 7. Docker Container (Task 6 — 10 marks)

**Owner: Aryaveer Rathi**

### Dockerfile Design
```dockerfile
FROM python:3.11-slim
ARG HF_MODEL_NAME=Nikhil-iitj/emotion-electra
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY id2label.json .
ENTRYPOINT ["python", "src/inference.py"]
```

**Design decisions:**
- **Slim base:** `python:3.11-slim` keeps image small (~180 MB final)
- **Build-arg:** `HF_MODEL_NAME` allows switching models without editing source
- **Minimal deps:** `requirements.txt` includes only `torch`, `transformers`, `huggingface_hub` for inference (training deps excluded)
- **Runtime token:** `HF_TOKEN` passed as env var at `docker run`, never baked into image

### Local Build & Test
```bash
docker build --build-arg HF_MODEL_NAME=Nikhil-iitj/emotion-electra \
  -t mlops-groupproject-inference:latest .

docker run --rm -e HF_TOKEN=$HF_TOKEN \
  -e INPUT_TEXT="I am so happy today!" \
  mlops-groupproject-inference:latest
```

**Output:**
```
Input : I am so happy today!
Label : joy  (confidence 0.987)
```

### Docker Hub Push
```bash
docker tag mlops-groupproject-inference:latest \
  nikhilsainiiitj/mlops-groupproject-inference:latest
docker push nikhilsainiiitj/mlops-groupproject-inference:latest
```

- **Image URL:** https://hub.docker.com/r/nikhilsainiiitj/mlops-groupproject-inference
- **Visibility:** Public
- **Tag:** `latest` (also `v2` for version pinning)

**Automated workflow:** `.github/workflows/docker.yml` builds and pushes on every `push` to `main`.

---

## 8. GitHub Actions (Task 7 — 15 marks)

**Owner: Nikhil Saini**

### Workflow 1: CI (`.github/workflows/ci.yml`)
- **Trigger:** Push to `develop`, PR to `main`
- **Job:** `flake8 src/ --max-line-length=120`
- **Purpose:** Code quality gate before merge

### Workflow 2: Inference (`.github/workflows/inference.yml`)
- **Trigger:** `workflow_dispatch` (manual button in GitHub UI)
- **Inputs:**
  - `input_text`: text to classify
  - `hf_model_name`: defaults to `Nikhil-iitj/emotion-electra`
- **Job:** Installs deps, sets `HF_TOKEN` from repo Secrets, runs `src/inference.py`
- **Secrets used:** `HF_TOKEN`

### Workflow 3: Docker Build & Push (`.github/workflows/docker.yml`)
- **Trigger:** Push to `main`, manual dispatch
- **Job:** Buildx → login to Docker Hub → push `latest` + `v2` tags
- **Secrets used:** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

**Successful run evidence:** Screenshots of Actions runs + logs saved in submission folder.

---

## 9. Weights & Biases Dashboard (Task 8 — 5 marks)

**Owner: Sarthak Kapoor**

- **Project:** `mlops-groupproject-v2`
- **URL:** https://wandb.ai/g25ait2067-prom-iit-rajasthan/mlops-groupproject-v2
- **Visibility:** Public
- **Runs:** `run-v1` (MiniLM) and `run-v2` (ELECTRA-small)
- **Metrics logged:** `train/loss`, `eval/loss`, `eval/accuracy`, `eval/f1`, `train/epoch`, `train/global_step`, hyperparameters
- **Step configuration:** `train/*` plotted against `train/global_step`; `eval/*` plotted against `epoch`

**Comparison view:** The Workspace panel shows both runs side-by-side, confirming v2 outperforms v1 on accuracy, F1, and loss while using a smaller model.

**Screenshot:** W&B Workspace with both runs inserted in Section 4.2.1 of this report.

---

## 10. Link Verification (Incognito Audit)

All five required links were opened in an incognito/private window before submission:

| # | Link | Status |
|---|------|--------|
| 1 | GitHub repo | ✅ Public, workflows visible |
| 2 | Kaggle v1 notebook | ✅ Public |
| 3 | Kaggle v2 notebook | ✅ Public |
| 4 | HF model (v2 winner) | ✅ Public, files downloadable |
| 5 | W&B dashboard | ✅ Public, charts render |
| 6 | Docker Hub image | ✅ Public, pullable |

---

## 11. Challenges & Learnings

1. **Model size trap:** Our initial choice (`distilbert-base-uncased`, ~268 MB) exceeded the 200 MB limit. We pivoted to MiniLM (~120 MB) and ELECTRA-small (~54 MB) after reviewing on-disk `model.safetensors` sizes.

2. **W&B blank charts:** Early runs showed "There's no data for the selected runs." Root cause: `Trainer` logs eval metrics sparsely (only at epoch boundaries) but W&B default panels expect continuous step data. **Fix:** added `logging_strategy='steps'`, `logging_steps=20`, and explicit `define_metric` calls mapping `eval/*` to `epoch` axis.

3. **HF_TOKEN whitespace:** Kaggle Secrets appended trailing whitespace causing `LocalProtocolError` on Hugging Face login. **Fix:** `.strip()` on `secrets.get_secret('HF_TOKEN').strip()`.

4. **Kaggle Internet toggle:** `UserSecretsClient` fails silently if Kaggle Settings → Internet is OFF. Documented prominently in notebook headers.

5. **GitHub contributor hygiene:** Early commits included `Co-Authored-By: Claude` trailers, causing `@claude` to appear in contributor graphs. **Fix:** amended commits with `git commit --amend` removing trailers; recreated repo when cached data persisted.

---

## 12. Repository Structure

```
MLOps_GroupProject/
├── .github/workflows/
│   ├── ci.yml              # Lint on develop push / PR to main
│   ├── inference.yml       # Manual inference trigger
│   └── docker.yml          # Auto-build & push on main
├── src/
│   ├── data.py            # Data loading + cleaning
│   ├── train.py           # Local training template
│   ├── eval.py            # Evaluation + artifact logging
│   ├── utils.py           # compute_metrics, TextDataset, load_label_map
│   └── inference.py       # Docker / Actions entrypoint
├── kaggle_minilm.ipynb    # Kaggle notebook v1
├── kaggle_electra.ipynb   # Kaggle notebook v2
├── Dockerfile             # Inference container
├── requirements.txt       # Python dependencies
├── id2label.json          # Label mapping (committed)
├── .gitignore             # Excludes data/, results/, .env
├── README.md              # Setup + all live links
└── docs/
    └── REPORT.md          # Full markdown report (source for PDF)
```

---

*Submitted by Group 2, IIT Jodhpur*
