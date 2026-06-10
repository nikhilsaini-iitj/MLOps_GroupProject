# MLOps Group Project — Team Decision Document
## Dataset + Model Selection

**Assignment Constraints (Hard Limits):**
- Dataset: reasonably small, **< 50,000 samples**
- Model: compact, **< 200 MB** on Hugging Face
- Training: must complete within **Kaggle free GPU T4 × 2** (~30 hrs/week)
- Modality: any — text, image, tabular, audio
- Task: well-scoped classification (text sentiment, image class, etc.)

---

## ⚠️ Important: Size Check

Do **not** assume any model is under 200 MB without checking.
- `distilbert-base-uncased` `model.safetensors` = **~268 MB** → **FAILS** the hard limit
- Always verify by opening the model card → Files and versions → check `.safetensors` or `.bin` size.

---

## Quick-Start Options (Pre-Vetted)

### Option A: Emotion Detection (Text) — RECOMMENDED

| Attribute | Value |
|-----------|-------|
| **Dataset** | `emotion` (Hugging Face `datasets`) |
| **Classes** | 6 — sadness, joy, love, anger, fear, surprise |
| **Size** | ~16,000 train / 2,000 val / 2,000 test |
| **v1 Model** | `microsoft/MiniLM-L12-H384-uncased` |
| **v1 Size** | ~33M params, **~120 MB** ✅ |
| **v2 Model** | `google/electra-small-discriminator` |
| **v2 Size** | ~14M params, **~54 MB** ✅ |
| **Training Time (each)** | ~8–12 min (3 epochs, Kaggle T4) |

**Why this works:**
- Both models are unambiguously under 200 MB
- Dataset loads in 1 line — no manual download
- Fast training = more time for Docker, Actions, report
- Using **two different models** (MiniLM vs ELECTRA-small) as v1/v2 gives a genuine size-vs-accuracy comparison for the report
- Text classification is the most straightforward modality for this assignment

**Report advantage:**
You can write a paragraph comparing MiniLM (distilled from 12-layer BERT, strong accuracy) vs ELECTRA-small (discriminator pretraining, very fast). This is a real architectural comparison, not just hyperparameter tweaking.

---

### Option B: AG News Classification (Text)

| Attribute | Value |
|-----------|-------|
| **Dataset** | `ag_news` (Hugging Face `datasets`) |
| **Classes** | 4 — World, Sports, Business, Sci/Tech |
| **Size** | ~120,000 train / 7,600 test |
| **v1 Model** | `microsoft/MiniLM-L12-H384-uncased` (~120 MB) |
| **v2 Model** | `google/electra-small-discriminator` (~54 MB) |
| **Training Time** | ~20 min (sampled to 30K) |

**Why this works:**
- Real-world news classification task
- Plenty of data cleaning to justify (deduplication, length filtering, class balancing)

**Risks:**
- Full dataset is 120K — must subsample to <50K
- Slightly longer training time

---

### Option C: Food-101 Image Classification (Image)

| Attribute | Value |
|-----------|-------|
| **Dataset** | `food101` subset (Hugging Face `datasets`) |
| **Classes** | 10 (subset of 101) |
| **Size** | ~7,500 images |
| **Model** | `microsoft/resnet-50` (~98 MB) ✅ |
| **Training Time** | ~20–30 min (3 epochs, Kaggle T4) |

**Why this works:**
- Different modality = stands out
- ResNet-50 is unambiguously under 200 MB

**Risks:**
- Image training uses more GPU memory — may need batch size 4–8
- Docker inference requires image input handling (more complex than text)
- Only pick if a teammate has prior image-classification experience

---

## Decision Matrix

| Factor | Option A (Emotion) | Option B (AG News) | Option C (Food-101) |
|--------|-------------------|-------------------|---------------------|
| **Ease of data loading** | ⭐⭐⭐ Very Easy | ⭐⭐ Easy | ⭐ Moderate |
| **Training speed** | ⭐⭐⭐ Fast (~10 min/run) | ⭐⭐ Medium (~20 min) | ⭐⭐ Medium (~25 min) |
| **Model < 200 MB** | ⭐⭐⭐ Yes (MiniLM 120 MB, ELECTRA 54 MB) | ⭐⭐⭐ Yes | ⭐⭐⭐ Yes (ResNet-50) |
| **Data cleaning to justify** | ⭐⭐ Some | ⭐⭐⭐ Plenty | ⭐⭐⭐ Plenty |
| **Docker simplicity** | ⭐⭐⭐ Text = easy | ⭐⭐⭐ Text = easy | ⭐ Image = harder |
| **v1/v2 differentiation** | ⭐⭐⭐ Two different models | ⭐⭐⭐ Two different models | ⭐⭐ Same model, diff hyperparams |
| **Risk of failure** | ⭐⭐⭐ Very Low | ⭐⭐ Low | ⭐ Higher |
| **Overall marks ease** | ⭐⭐⭐ Highest | ⭐⭐⭐ High | ⭐⭐ Medium |

---

## My Recommendation

**Option A (Emotion + MiniLM v1 / ELECTRA-small v2)** is the safest and smartest choice:
- Both models are clearly under 200 MB
- Fastest training = maximum time for Docker, Actions, report
- Using **two different models** as v1 and v2 gives a real comparison to discuss (not just "batch size 16 vs 32")
- Lowest risk of GPU OOM or Kaggle timeout

**Only pick Option C if** your team wants to differentiate and at least one member has prior image-classification experience.

---

## Team Vote

Reply with your name and choice (A / B / C). Majority wins.

| Name | Roll Number | Choice (A/B/C) | Reason (optional) |
|------|-------------|----------------|-------------------|
| Nikhil Saini | G25AIT2067 | **A** | Fastest pipeline, two-model comparison, under 200 MB |
| Y Sharathchandrika | G25AIT2132 | | |
| Sarthak Kapoor | G25AIT2098 | | |
| Aryaveer Rathi | G25AIT2021 | | |

**Deadline for vote:** *(suggest within 24 hours)*

---

## Next Steps After Decision

Once we pick an option, work will be divided:

| Task | Owner | Phase |
|------|-------|-------|
| Data prep + id2label.json | TBD | Phase 2 |
| Kaggle notebook v1 (MiniLM training) | TBD | Phase 4 |
| Kaggle notebook v2 (ELECTRA-small training) | TBD | Phase 4 |
| Docker container + push | TBD | Phase 7 |
| GitHub Actions setup + Secrets | TBD | Phase 8 |
| Report writing | TBD | Phase 10 |
| W&B dashboard screenshot | TBD | Phase 9 |

---

*Document updated with corrected model sizes (DistilBERT removed — over 200 MB limit).*
*Created by Nikhil Saini (G25AIT2067) for Group 2 — MLOps End-to-End Pipeline, IIT Jodhpur*
