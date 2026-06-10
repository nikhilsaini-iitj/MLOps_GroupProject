# MLOps Group Project — Execution Plan
## End-to-End Pipeline (100 Marks)

---

## 1. Project Scope & Choices

### Dataset: Emotion (Hugging Face Datasets)
- **Source:** `datasets.load_dataset("emotion")`
- **Classes:** 6 (sadness, joy, love, anger, fear, surprise)
- **Size:** ~16,000 train / ~2,000 validation / ~2,000 test
- **Modality:** Text classification
- **Why:** Small (<50K), loads instantly, no manual download, well-known benchmark

### Model: distilbert-base-uncased
- **Size:** 66M parameters (~250 MB)
- **Why:** Under 200 MB, trains in ~15 min on Kaggle T4, strong baseline for text
- **HF Card:** https://huggingface.co/distilbert-base-uncased

### Task: Emotion Detection from Short Text
- Input: tweet-length text
- Output: one of 6 emotion labels

---

## 2. Execution Phases

### Phase 1: Repository Setup (10 marks)
- [x] Repo created: `nikhilsaini-iitj/MLOps_GroupProject`
- [x] `main` + `develop` branches
- [x] Branch protection on `main` (requires 1 PR review)
- [ ] Add group members as collaborators (WRITE access)
- [ ] Screenshot collaborator settings for report

### Phase 2: Data Preparation (15 marks)
- [ ] Load `emotion` dataset via `datasets` library
- [ ] Inspect: class distribution, sample lengths, duplicates
- [ ] Clean: lowercase, strip extra whitespace, remove exact duplicates
- [ ] Encode labels → save `id2label.json`
- [ ] Stratified train/val/test split
- [ ] Commit ONLY `id2label.json` + scripts (not data files)
- **Deliverable:** `src/data.py` fully implemented

### Phase 3: Model Selection (10 marks)
- [ ] Load `distilbert-base-uncased` tokenizer
- [ ] Load model with `num_labels=6`
- [ ] Write 100–150 word justification referencing HF model card
- **Deliverable:** `src/utils.py` updated + report section

### Phase 4: Kaggle Training — 2 Versions (25 marks)
- [ ] Import notebook into Kaggle, enable GPU T4 x2
- [ ] Add Kaggle Secrets: `WANDB_API_KEY`, `HF_TOKEN`
- [ ] **Version 1 (v1):** epochs=3, batch_size=16, lr=5e-5
- [ ] **Version 2 (v2):** epochs=3, batch_size=32, lr=2e-5
- [ ] Log all metrics to W&B project `mlops-groupproject`
- [ ] Screenshot W&B dashboard showing both runs side-by-side
- **Deliverable:** 2 public Kaggle notebooks + W&B dashboard

### Phase 5: Evaluation & Artifacts (5 marks)
- [ ] Run final eval on best version
- [ ] Log accuracy, F1, loss to W&B
- [ ] Save `classification_report` as JSON
- [ ] Upload eval report as W&B Artifact
- **Deliverable:** `src/eval.py` + artifact on W&B

### Phase 6: Hugging Face Deployment (5 marks)
- [ ] Push best model + tokenizer to HF Hub
- [ ] Repo name: `Nikhil-iitj/distilbert-emotion`
- [ ] Log HF URL in W&B run summary
- **Deliverable:** Public HF model repo

### Phase 7: Docker Container (10 marks)
- [ ] `Dockerfile` using `python:3.11-slim`
- [ ] `ARG HF_MODEL_NAME` with default
- [ ] Install only inference deps (no training libs)
- [ ] Build locally and test: `docker run -e HF_TOKEN -e INPUT_TEXT ...`
- [ ] Push to Docker Hub: `nikhilsaini/mlops-groupproject-inference`
- **Deliverable:** Public Docker image URL

### Phase 8: GitHub Actions (15 marks)
- [ ] **CI workflow** (`ci.yml`): triggers on push to `develop` — flake8 lint
- [ ] **Inference workflow** (`inference.yml`): manual trigger with `INPUT_TEXT` + `HF_MODEL_NAME`
- [ ] Add GitHub Secrets: `HF_TOKEN` (Settings → Secrets → Actions)
- [ ] Run inference workflow, capture successful run screenshot/badge
- **Deliverable:** 2 workflow files + successful run log

### Phase 9: W&B Experiment Comparison (5 marks)
- [ ] Ensure both v1 and v2 appear in W&B project
- [ ] Set project visibility to PUBLIC
- [ ] Use Runs Comparison table in report
- **Deliverable:** Public W&B URL + comparison table in report

### Phase 10: Report (5 marks)
- [ ] 4–5 page PDF with all required sections
- [ ] Collaborator roles screenshot
- [ ] Data cleaning decisions
- [ ] Model selection rationale (HF model card ref)
- [ ] Experiment comparison table (v1 vs v2)
- [ ] GitHub Actions successful run screenshot
- [ ] Dockerfile design explanation + Actions log
- [ ] Challenges & learnings
- [ ] All live links (GitHub, 2×Kaggle, HF, Docker, W&B)
- **Deliverable:** `report.pdf` uploaded to portal

---

## 3. Hyperparameter Comparison Table

| Parameter | Version 1 (v1) | Version 2 (v2) |
|-----------|----------------|----------------|
| Epochs | 3 | 3 |
| Batch size | 16 | 32 |
| Learning rate | 5e-5 | 2e-5 |
| Warmup steps | 100 | 100 |
| Weight decay | 0.01 | 0.01 |
| Expected time | ~15 min | ~10 min |

---

## 4. File Map

```
MLOps_GroupProject/
├── .github/workflows/
│   ├── ci.yml
│   └── inference.yml
├── src/
│   ├── data.py          ← Phase 2
│   ├── utils.py          ← Phase 3
│   ├── train.py          ← Phase 4 (local template)
│   ├── eval.py           ← Phase 5
│   └── inference.py      ← Phase 7/8
├── Dockerfile            ← Phase 7
├── requirements.txt      ← all phases
├── id2label.json         ← Phase 2
├── README.md             ← Phase 10
└── report.md / report.pdf ← Phase 10
```

---

## 5. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Kaggle GPU hours exhausted | Use small dataset (16K), 3 epochs max |
| HF push fails | Verify token is Write-scoped; use `.strip()` |
| Docker image too large | Use `python:3.11-slim`, no training deps |
| GitHub Actions inference fails | HF token in repo Secrets, not code |
| W&B not logging | `report_to="wandb"` in TrainingArguments |
| Group member commits | Admin adds collaborators immediately |

---

## 6. Timeline Estimate

| Phase | Estimated Time |
|-------|---------------|
| 1: Repo setup | Done |
| 2: Data prep | 30 min |
| 3: Model selection | 20 min |
| 4: Kaggle v1 training | 20 min (GPU) |
| 4: Kaggle v2 training | 15 min (GPU) |
| 5: Evaluation | 10 min |
| 6: HF push | 5 min |
| 7: Docker | 30 min |
| 8: GitHub Actions | 20 min |
| 9: W&B comparison | 10 min |
| 10: Report | 45 min |
| **Total** | **~3–4 hours** |

---

*Plan created for Nikhil Saini (G25AIT2067) — MLOps Group Project, IIT Jodhpur*
