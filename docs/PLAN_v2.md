# MLOps End-to-End Pipeline — Execution Plan (v2)
## Group 2 · PGD AI · IIT Jodhpur · 100 marks

Members: Nikhil Saini (G25AIT2067) · Y Sharathchandrika (G25AIT2132) · Sarthak Kapoor (G25AIT2098) · Aryaveer Rathi (G25AIT2021)

---

## 0. Locked decisions

| Item | Decision |
|------|----------|
| Task | Text classification — Emotion detection (6 classes) |
| Dataset | `dair-ai/emotion` (~16k train / 2k val / 2k test) |
| Model v1 | `microsoft/MiniLM-L12-H384-uncased` (~120 MB) |
| Model v2 | `google/electra-small-discriminator` (~54 MB) |
| Why two models | Satisfies "different configuration" + gives a real size-vs-accuracy story (distillation vs replaced-token-detection) |
| Training | Kaggle Notebook, GPU T4, HF `Trainer` API |
| Tracking | Weights & Biases (public project) |
| Registry | Docker Hub (public) |

> Grading note: **marks come from the pipeline, not accuracy.** Treat the model as a black box. Every decision must be justified in the report.

---

## 1. Role ownership (everyone must commit — history is graded)

| Member | Primary role | Owns tasks |
|--------|--------------|-----------|
| **Nikhil Saini** | Repo & DevOps lead | Task 1 (repo, branch protection, collaborators), Task 7 (both workflows, GitHub Secrets), report compile |
| **Y Sharathchandrika** | Data lead | Task 2 (inspect, clean, `id2label.json`, `data.py`) |
| **Sarthak Kapoor** | Training & W&B lead | Task 3 (load model/tokenizer), Task 4 (2 Kaggle runs + W&B), Task 5 (push to HF), Task 8 (public dashboard + comparison) |
| **Aryaveer Rathi** | Inference & Docker lead | `inference.py`, Task 6 (Dockerfile, build/test/push), supports Task 7 inference workflow |

Each member writes their **own report section** (their task + rationale); Nikhil merges into the final 4–5 page PDF.

---

## 2. Repo structure (commit empty placeholders Day 1 to avoid merge conflicts)

```
mlops-emotion/
├── README.md            # setup + all 5 live links
├── .gitignore           # ignore data/, results/, __pycache__, *.ckpt
├── LICENSE              # MIT
├── requirements.txt
├── id2label.json        # committed (mapping only, NOT the dataset)
├── src/
│   ├── data.py          # Sharathchandrika
│   ├── utils.py         # Shared helpers
│   ├── train.py         # Sarthak (also lives in Kaggle notebook)
│   ├── eval.py          # Sarthak
│   └── inference.py     # Aryaveer
├── Dockerfile           # Aryaveer
└── .github/workflows/
    ├── ci.yml           # Nikhil
    └── inference.yml    # Nikhil
```

---

## 3. Phased plan with dependencies

Critical path: **Repo → Data → Training → HF push → Inference/Docker → Actions → Report.**
Docker + Actions scaffolding can be built *in parallel* using any public placeholder model, then re-pointed to the real model once it's on HF.

### Phase 1 — Foundations (Day 1, parallel) · Tasks 1 & 2 start
- **Nikhil:** create public repo, add README/.gitignore/LICENSE, create `develop` branch, protect `main` (require 1 PR review), add the other 3 as Collaborators with Write, **screenshot the Collaborators page** → save for report. *(Task 1 = 10 marks)*
- **Everyone:** commit one placeholder file in your folder so all four show up in history immediately.
- **Sharathchandrika:** load `dair-ai/emotion`, print size/structure/class distribution → screenshot for report.

### Phase 2 — Data prep (Day 2) · Task 2 (15 marks)
- `data.py`: lowercase, strip punctuation, drop duplicates, check/handle nulls, note class imbalance (emotion is imbalanced — `joy`/`sadness` dominate; document this, don't silently fix).
- Encode labels → write `id2label.json` (and `label2id`). Commit **only** the mapping.
- Save processed splits locally (gitignored).
- Write the **data-cleaning rationale** report section now while it's fresh.

### Phase 3 — Model load + training script (Day 2–3) · Task 3 (10 marks)
- **Sarthak:** load tokenizer + `AutoModelForSequenceClassification` with `num_labels=len(id2label)`, `id2label`/`label2id` wired in.
- Write the **model-selection paragraph (100–150 words)** referencing the HF model card (params, distillation/RTD objective, why <200 MB).

### Phase 4 — Kaggle training, 2 versions (Day 3–4) · Task 4 (25 marks — biggest block)
- Add `WANDB_API_KEY` + `HF_TOKEN` as **Kaggle Secrets** (never hardcode).
- **Run v1:** MiniLM, 3 epochs, bs 16, lr 3e-5, `report_to='wandb'`, `run_name='run-v1'`.
- **Run v2:** ELECTRA-small, 3 epochs, bs 16, lr 5e-5, `run_name='run-v2'`.
- Log train loss, val loss, accuracy, F1, and all hyperparameters for both.
- **Two public Kaggle notebooks** (one per version) — set both to Public.
- Screenshot W&B dashboard with both runs.

### Phase 5 — Publish model (Day 4) · Task 5 (5 marks)
- Push best model **weights + tokenizer** to a **public** HF repo.
- `wandb.run.summary['huggingface_model'] = <url>`.
- **Verify the on-disk size** of the pushed checkpoint and record the real number:
  ```python
  import os; print(f"{os.path.getsize('model.safetensors')/1e6:.1f} MB")
  ```

### Phase 6 — Inference + Docker (Day 3–5, parallelisable) · Task 6 (10 marks)
- **Aryaveer:** `inference.py` reads `INPUT_TEXT` env var + `HF_MODEL_NAME`, prints predicted label. Build/test against a placeholder public model first, swap to real model after Phase 5.
- `Dockerfile`: slim Python base (`python:3.11-slim`), install **only** inference deps, `ARG HF_MODEL_NAME` with sensible default.
- Build + test locally end-to-end → push to Docker Hub (public). Save build/run logs for report.

### Phase 7 — GitHub Actions (Day 5) · Task 7 (15 marks)
- **Nikhil:** `ci.yml` (push to `develop` / PR to `main` → flake8 lint).
- `inference.yml` (`workflow_dispatch` with `input_text` → runs `inference.py` pulling the HF model).
- Add `HF_TOKEN` (and `WANDB_API_KEY` if used) as **GitHub repo Secrets**.
- Trigger a successful inference run → **screenshot/badge + paste the Actions log** for report.

### Phase 8 — W&B finalise (Day 5) · Task 8 (5 marks)
- Set W&B project visibility **Public**.
- Use Runs Comparison table showing Accuracy / F1 / Loss side-by-side → screenshot.

### Phase 9 — Report + link audit (Day 6) · Report (5 marks)
- Compile 4–5 page PDF: names + contributions, all 5 live links, required screenshots, rationale paragraphs, **v1-vs-v2 metrics table + which won and why**, Docker design notes + Actions log, challenges & learnings.
- **Link audit (do this last, together):** open every link in an incognito window — broken or private = **zero for that component**. Check: GitHub (public, workflows present), both Kaggle notebooks (public), HF model (public), Docker image (publicly pullable), W&B (public).

---

## 4. Critical path & risks

```
Task1 ─┐
       ├─► Task2 ─► Task3 ─► Task4 ─► Task5 ─► Task7(inference) ─► Report
Task6 ─┘ (scaffold early w/ placeholder model)        ▲
                                                       └ Task8 (W&B public)
```

| Risk | Mitigation |
|------|-----------|
| Model >200 MB (the DistilBERT trap) | MiniLM/ELECTRA both well under; verify actual `model.safetensors` size before submit |
| Kaggle GPU hours run low | Subsample, fewer epochs; emotion is small so low risk |
| A link is private at grading | Final incognito audit; HF + Docker + W&B + both Kaggle set Public |
| One member has no commits | Each owns code they personally commit; check history before submit |
| Tokens leaked in commits | Kaggle Secrets + GitHub Secrets only; `.gitignore` + scan diff before push |

---

## 5. Definition of done (pre-submission checklist)

- [ ] `main` protected, 1-review rule, 3 collaborators with Write, screenshot saved
- [ ] `id2label.json` committed; raw/processed data **not** committed
- [ ] Model-selection paragraph cites the HF model card
- [ ] 2 Kaggle runs differ in ≥1 hyperparameter, both Public, both log loss/acc/F1 + hyperparams
- [ ] Best model + tokenizer on public HF; URL in W&B summary
- [ ] Dockerfile uses slim base + `ARG HF_MODEL_NAME`; image publicly pullable
- [ ] `ci.yml` + `inference.yml` present; successful inference run screenshot + log
- [ ] W&B project Public with comparison table
- [ ] 4–5 page PDF with all 5 live links verified in incognito
- [ ] All 4 members appear in commit history
