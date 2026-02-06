# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** The Lord of the Git

## Executive Summary
This project implements a multimodal, ensemble pricing solution designed to predict product prices based on heterogeneous inputs: text descriptions, product images, and derived tabular metadata. By combining the strengths of transformer-based embeddings (using `sentence-transformers` and Vision Transformers) with robust ensemble regressors (XGBoost and Ridge), our solution achieves accurate price predictions even for complex or sparse data entries.

## Key Features
- **Multimodal Analysis:** Integrates text, image, and tabular data for holistic product understanding.
- **Advanced Embeddings:** 
  - Text: `sentence-transformers/all-distilroberta-v1`
  - Images: `google/vit-base-patch16-224`
- **Ensemble Learning:** Uses a stacked ensemble approach combining XGBoost (nonlinear) and Ridge Regression (linear) for robust predictions.
- **Robustness:** Handles missing images and sparse text descriptions effectively.

## Prerequisites
- Python 3.8+
- Recommended: GPU support for faster Transformer inference.

## Installation

1. **Clone the repository** (if applicable) or download the project files.
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
- **`Amazon ML Challenge.ipynb`**: The main Jupyter Notebook containing the end-to-end pipeline:
    - Data loading and preprocessing
    - Text and Image embedding extraction
    - Feature engineering
    - Model training (XGBoost, Ridge, Stacking)
    - Inference and evaluation
- **`Documentation.md`**: Detailed technical documentation describing the methodology, architecture, and analyses.
- **`submission.csv`**: Generated file containing the model's price predictions.
- **`requirements.txt`**: List of Python dependencies.

## Usage

1. Open the main notebook:
   ```bash
   jupyter notebook "Amazon ML Challenge.ipynb"
   ```
2. **Run the cells sequentially.**
   - The notebook relies on `train.csv` (ensure this dataset is present in the directory).
   - It will download images using the links provided in the dataset. **Note:** Image downloading may take some time depending on your connection.
   - The final cells generate the `submission.csv` file.

## Methodology Overview

### 1. Feature Extraction
- **Text:** We explicitly parse structured information (Brand, Pack Size, Units) from the raw text and use `all-distilroberta-v1` to generate dense semantic embeddings.
- **Images:** We download images, resize them to 224x224, and extract visual features using a pre-trained Vision Transformer (ViT).
- **Tabular:** We engineer features such as title length, presence of specific keywords (e.g., "organic", "refill"), and currency symbols to capture structured signals.

### 2. Model Architecture
The solution uses a **stacking ensemble**:
1. **Base Learners:** Independent XGBoost and Ridge Regression models trained on the concatenated feature vector (Text Embeddings + Image Embeddings + Tabular Features).
2. **Meta Learner:** A final estimator that combines the predictions from the base learners to produce the final price. This allows the model to capture both linear trends and complex non-linear interactions.

## Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities and Ridge regression
- `xgboost` - Gradient boosted decision trees
- `lightgbm` - Gradient boosting framework
- `transformers` - Hugging Face model utilities
- `sentence-transformers` - Text embedding generation
- `Pillow` - Image processing
- `tqdm` - Progress bars
- `requests` - Image downloading

## License
MIT License
