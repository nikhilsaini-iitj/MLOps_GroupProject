# MLOps Group Project — Team Decision Document
## Dataset + Model Selection

**Assignment Constraints (Hard Limits):**
- Dataset: reasonably small, **< 50,000 samples**
- Model: compact, **< 200 MB** on Hugging Face
- Training: must complete within **Kaggle free GPU T4 × 2** (~30 hrs/week)
- Modality: any — text, image, tabular, audio
- Task: well-scoped classification (text sentiment, image class, etc.)

---

## Quick-Start Options (Pre-Vetted)

We have shortlisted **3 options** that meet all constraints. Review, discuss, and vote below.

---

### Option A: Emotion Detection (Text)

| Attribute | Value |
|-----------|-------|
| **Dataset** | `emotion` (Hugging Face `datasets`) |
| **Classes** | 6 — sadness, joy, love, anger, fear, surprise |
| **Size** | ~16,000 train / 2,000 val / 2,000 test |
| **Model** | `distilbert-base-uncased` |
| **Model Size** | 66M params (~255 MB) |
| **Training Time** | ~12–15 min (3 epochs, Kaggle T4) |

**Why this works:**
- Dataset loads in 1 line (`datasets.load_dataset("emotion")`)
- No manual download or preprocessing headaches
- DistilBERT trains fast, strong baseline, well-documented model card
- Text classification is the most straightforward modality for this assignment

**Risks:**
- Common choice — many other groups may pick the same
- Must justify cleaning decisions even though data is already fairly clean

---

### Option B: AG News Classification (Text)

| Attribute | Value |
|-----------|-------|
| **Dataset** | `ag_news` (Hugging Face `datasets`) |
| **Classes** | 4 — World, Sports, Business, Sci/Tech |
| **Size** | ~120,000 train / 7,600 test |
| **Model** | `distilbert-base-uncased` |
| **Model Size** | 66M params (~255 MB) |
| **Training Time** | ~25–30 min (sample to 30K) or ~45 min (full) |

**Why this works:**
- Real-world news classification task
- Model card justification is easy (DistilBERT proven on news)
- Can demonstrate clear data cleaning (deduplication, length filtering)

**Risks:**
- Full dataset is 120K — must subsample to <50K to stay within assignment limits
- Slightly longer training time

---

### Option C: Food-101 Image Classification (Image)

| Attribute | Value |
|-----------|-------|
| **Dataset** | `food101` subset (Hugging Face `datasets`) |
| **Classes** | 10 (subset of 101) — e.g., pizza, steak, sushi |
| **Size** | ~7,500 images (750 per class) |
| **Model** | `google/vit-base-patch16-224` OR `microsoft/resnet-50` |
| **Model Size** | ~86M–25M params (~330 MB / ~100 MB) |
| **Training Time** | ~20–30 min (3 epochs, Kaggle T4) |

**Why this works:**
- Different modality = stands out in class
- Good demonstration of image preprocessing (resize, normalise, augment)
- ResNet-50 is under 200 MB and proven on images

**Risks:**
- Image training uses more GPU memory — may need batch size 8 or 4
- `datasets` image loading can be slower; may need manual download
- Docker inference requires image input handling (more complex than text)
- If using ViT, model is ~330 MB (over limit) — must use ResNet or smaller ViT variant

---

## Decision Matrix

| Factor | Option A (Emotion) | Option B (AG News) | Option C (Food-101) |
|--------|-------------------|-------------------|---------------------|
| **Ease of data loading** | ⭐⭐⭐ Very Easy | ⭐⭐ Easy | ⭐ Moderate |
| **Training speed** | ⭐⭐⭐ Fast (~15 min) | ⭐⭐ Medium (~30 min) | ⭐⭐ Medium (~25 min) |
| **Model < 200 MB** | ⭐⭐⭐ Yes (255 MB on disk, 66M params) | ⭐⭐⭐ Yes | ⭐⭐ Only if ResNet-50 |
| **Data cleaning to justify** | ⭐⭐ Some | ⭐⭐⭐ Plenty | ⭐⭐⭐ Plenty |
| **Docker simplicity** | ⭐⭐⭐ Text = easy | ⭐⭐⭐ Text = easy | ⭐ Image = harder |
| **Uniqueness / standout** | ⭐ Common | ⭐⭐ Less common | ⭐⭐⭐ Rare |
| **Risk of failure** | ⭐⭐⭐ Low | ⭐⭐ Low | ⭐ Higher |
| **Overall marks ease** | ⭐⭐⭐ High | ⭐⭐⭐ High | ⭐⭐ Medium |

---

## My Recommendation

**Option A (Emotion + DistilBERT)** is the safest choice:
- Fastest to implement = more time for Docker, Actions, report
- Lowest risk of GPU OOM or timeout
- Still allows full marks if we justify cleaning steps and model choice well

**Only pick Option C if** your team wants to differentiate and at least one member has prior image-classification experience.

---

## Team Vote

Reply with your name and choice (A / B / C). Majority wins.

| Name | Roll Number | Choice (A/B/C) | Reason (optional) |
|------|-------------|----------------|-------------------|
| Nikhil Saini | G25AIT2067 | **A** | Fastest pipeline, lowest risk, more time for Docker/Actions |
| Y Sharathchandrika | G25AIT2132 | | |
| Sarthak Kapoor | G25AIT2098 | | |
| Aryaveer Rathi | G25AIT2021 | | |

**Deadline for vote:** *(fill in — suggest within 24 hours so we can start)*

---

## Next Steps After Decision

Once we pick an option, we will divide the work:

| Task | Owner | Phase |
|------|-------|-------|
| Data prep + id2label.json | TBD | Phase 2 |
| Kaggle notebook v1 (training) | TBD | Phase 4 |
| Kaggle notebook v2 (different hyperparams) | TBD | Phase 4 |
| Docker container + push | TBD | Phase 7 |
| GitHub Actions setup + Secrets | TBD | Phase 8 |
| Report writing | TBD | Phase 10 |
| W&B dashboard screenshot | TBD | Phase 9 |

---

*Document created by Nikhil Saini (G25AIT2067) for Group 2 — MLOps End-to-End Pipeline, IIT Jodhpur*
