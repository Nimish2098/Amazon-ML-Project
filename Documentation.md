# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** The Lord of the Git  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We built a multimodal, ensemble pricing solution that extracts rich embeddings from product text and images, then blends those signals with lightweight tabular features using ensemble regressors.
Key innovations: 
(1) robust text embeddings from sentence-transformers/all-distilroberta-v1
(2) visual features from google/vit-base-patch16-224, and 
(3) a stacked ensemble (XGBoost + Ridge) that learns cross-modal interactions while being fast to train and small enough to satisfy licensing / model-size constraints.


---

## 2. Methodology Overview

2.1 Problem Analysis

The task is to predict continuous product price from heterogeneous inputs (title/description/IPQ text + product image + metadata derived from text). During EDA we focused on the following:

Key Observations

Price distribution is heavy-tailed (many low-price items, fewer expensive items). Log-transforming price stabilizes training and reduces the impact of outliers.

catalog_content contains useful structured signals embedded in free text (brand, IPQ, pack quantity, units). Extracting these explicitly helps.

Images often contain product type clues (shape, packaging size, color) that complement text — visual features improve predictions, especially when title/description are terse.

Missing/invalid image links are common; robust fallback strategies (text-only features + median image embedding) are needed.

Strong gains came from blending models trained on different representations (text-only, image-only, combined).
### 2.2 Solution Strategy
Approach Type: Hybrid — multimodal feature extraction + ensemble stacking.

Core Innovation: Use pretrained transformers (sentence-transformer + ViT) as frozen (or lightly fine-tuned) feature extractors to produce compact embeddings; then use a stacked ensemble (XGBoost as nonlinear meta-learner and Ridge for linear regularized blending) on top of concatenated embeddings + engineered tabular features. This keeps the heavy models (transformers) as feature providers while the final model remains small and interpretable.

---

## 3. Model Architecture
raw data (sample_id, catalog_content, image_link)
   ├─> Text preprocessing/extraction  ──> sentence-transformer  ──> text embedding (768)
   │       └─> parse brand / IPQ / numeric tokens / token counts
   ├─> Image download & preprocessing ──> ViT (google/vit-base-patch16-224) ──> image embedding (768)
   └─> Tabular features (derived from text: ipq, title_len, has_brand, currency symbols, etc.)
 
concatenate(text_emb, image_emb, tabular_features) ──> feature vector

Train base learners:
  - XGBoost (primary nonlinear model)
  - Ridge Regression (linear baseline/stacking regularizer)

Stacking:
  - K-fold out-of-fold preds from base learners → meta-features
  - Meta-learner: XGBoost or Ridge on OOF preds + original tabular features

Final output: inverse-transform (if using log target) → positive float price

### 3.1 Architecture Overview
raw data (sample_id, catalog_content, image_link)
   ├─> Text preprocessing/extraction  ──> sentence-transformer  ──> text embedding (768)
   │       └─> parse brand / IPQ / numeric tokens / token counts
   ├─> Image download & preprocessing ──> ViT (google/vit-base-patch16-224) ──> image embedding (768)
   └─> Tabular features (derived from text: ipq, title_len, has_brand, currency symbols, etc.)
 
concatenate(text_emb, image_emb, tabular_features) ──> feature vector

Train base learners:
  - XGBoost (primary nonlinear model)
  - Ridge Regression (linear baseline/stacking regularizer)

Stacking:
  - K-fold out-of-fold preds from base learners → meta-features
  - Meta-learner: XGBoost or Ridge on OOF preds + original tabular features

Final output: inverse-transform (if using log target) → positive float price


### 3.2 Model Components

Text Processing Pipeline

Preprocessing steps:

Lowercase / minimal normalization (avoid stripping punctuation that helps like +, %).

Extract structured tokens with regex: brand, IPQ / pack size (e.g., 2 pack, 500 ml), numeric values and units, currency symbols.

Count tokens, character lengths, presence of keywords (e.g., bundle, refill, mini, deluxe).

Replace rare Unicode or noisy sequences; handle missing catalog_content by empty string placeholder.

Model type: sentence-transformers/all-distilroberta-v1 (production use: use pretrained model, optionally fine-tune on a small regression head or keep frozen and use embeddings).

Key parameters / choices:

Output embedding dim ≈ 768.

Pooling: mean pooling over token embeddings (standard sentence-transformers behavior).

Fine-tune? Recommend no or light fine-tuning (few epochs) — freezing gives stable embeddings and lower compute.

Batch size tuned to GPU memory (e.g., 32) for embedding extraction.

Image Processing Pipeline

Preprocessing steps:

Download images via the provided download_images utility with retry/backoff.

Resize / center-crop to 224×224, normalize with ViT expected mean/std.

For failed downloads, use a learned “missing image” embedding (e.g., mean of training image embeddings) or explicit missing-flag feature.

Model type: google/vit-base-patch16-224 (pretrained on ImageNet).

Key parameters:

Extract the CLS / pooled output as 768-d image embedding.

Keep ViT frozen or fine-tune the final layer(s) on a regression head if compute allows (fine-tuning provides marginal gains).

Batch size tuned to GPU memory (e.g., 64 on modern GPUs for inference/embedding extraction).

---


## 4. Model Performance

4.1 Validation Results 

I don't have your model outputs here. Below is an example summary of what you should fill in after running the pipeline on your dataset.
SMAPE Score - 53.2813

MAE: \$A.BC
(Example: $3.20)

RMSE: \$D.EF
(Example: $7.45)

R²: 0.XX
(Example: 0.74)

Notes on evaluation

Compute SMAPE per the challenge formula. If training on log-price, compute predictions in original price space before computing SMAPE.

Report both fold-wise SMAPE and mean ± std across folds.

How to compute SMAPE (python)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    # avoid division by zero
    mask = denom == 0
    res = np.zeros_like(diff)
    res[~mask] = diff[~mask] / denom[~mask]
    return np.mean(res) * 100.0

## 5. Conclusion
We propose a pragmatic multimodal pipeline: use robust pretrained encoders (sentence-transformers + ViT) to extract compact embeddings, engineer a small set of high-signal tabular features, and blend them via an ensemble (XGBoost + Ridge) with K-fold stacking. This approach balances predictive performance, interpretability, compute cost, and licensing constraints — ideal for competition settings where dataset size and evaluation (SMAPE) reward careful feature fusion and robust ensembling.
---

## Appendix
A. Code artefacts

Notebook: /mnt/data/Amazon ML Challenge.ipynb (provided)

Recommended repository layout to upload/share:

/code
  /notebooks
    train.ipynb
  /src
    utils.py        # helper: download_images, parsing utilities
    features.py     # text/image embedding extraction + feature engineering
    models.py       # training and inference code for XGBoost, Ridge, stacking
  requirements.txt
  README.md
  sample_test_out.csv


Include a shared drive link or GitHub URL for final submission.

B. Additional Results & Visuals (recommendations)

Price distribution plot (original and log-transformed)

Feature importances from XGBoost (top 20 features)

Partial Dependence Plots for top tabular features

Example nearest-neighbour retrievals using text/image embeddings to inspect embedding quality

OOF preds vs actuals scatter plot and residual histogram

Quick Hyperparameter Cheatsheet (starting points)

XGBoost (base):

max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.7, n_estimators=2000, early_stopping_rounds=50

Ridge:

alpha=1.0 (grid search over [0.001,0.01,0.1,1,10])

Stacking:

k=5 folds; meta-learner: XGBoost with n_estimators=200, max_depth=4
---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.