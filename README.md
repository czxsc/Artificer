# Artificer
AI-Powered Art History App that classifies paintings and retrieves relevant historical context

## Current Foundation Workflow

Install the Phase 0/1 Python dependencies with `pip install -r requirements.txt`.

### Phase 0

- Audit WikiArt labels and dataset imbalance with `python src/dataset_analysis.py`
- Build the clean Phase 0 datasets with `python src/setup_phase0_datasets.py`
- Default behavior removes `Unknown Artist` rows and keeps `Unknown Genre`
- This writes `train`, `val`, `known_artworks_test`, and `degraded_known_artworks` manifests to `outputs/phase0/`
- Materialize degraded test images with `python src/materialize_degraded_test_set.py`

### Phase 1

- Train the first frozen-CLIP baseline with `python src/train_clip_classifier.py`
- The baseline uses `openai/clip-vit-base-patch32` image embeddings and separate heads for artist, genre, and style
- Outputs are written to `outputs/phase1/`

# Phase 1: Which Vision Architecture Best Understand Paintings

**Objective:** Evaluate whether foundation-model representations (CLIP) or fully fine-tuned vision models (ResNet/ViT) are better at recognizing artistic characteristics from paintings.

### Dataset

We will be using the Huggingface Wikiart dataset (~81, 000 works with artist, genre, and style labels). https://huggingface.co/datasets/huggan/wikiart

Before training, first check

- Identify all the possible artist, genre, and style labels
- Output the number/distribution between all the labels. Sort and display all the labels as well as the number of works with that label
- Identify biases/categories with larger weights

Splitting data into 80 : 10 : 10 for train, validation, test → stratified by label

### Models

**CLIP** 

- Zero shot (Classification via prompt matching? Mostly for foundational baseline)
- CLIP + Linear Head (Frozen CLIP → Embedding → Linear Layer)
- CLIP + MLP (Frozen CLIP → Embedding → MLP)

**ResNet50** → Cnn baseline. Pretrained, then fine-tune 

**ViT →** Fine tuned vision transformer

### Evaluation

- Style Classification (Accuracy, F1, Top3/5 Accuracy)
- Genre Classification (Accuracy, F1, Top3/5 Accuracy)
- Artist Classification (Accuracy, F1, Top3/5 Accuracy)
- Check for overfitting/underfitting/convergence speed, Track:
    - Training loss
    - Validation loss
    - Training accuracy
    - Validation accuracy
    
    Plots:
    
    - Loss curves
    - Accuracy curves
- Robustness Benchmark (Blurs, crop, lighting changes, perspective distortion, tilts, etc.)
    - Report relative accuracy drop for each corruption type
- Error analysis of confusion matrix (which artist/styles/genres are confused most often?)
- Per-Class Performance (F1) Analysis

#### **Scope Limitations**

- Paintings only (Leave out artifacts and sculptures)
- Leave out matching/retrieval (dont worry about returning the exact work first, we first just produce correct labels and classifications)
- Limit to just the WikiArt Huggingface Dataset
- Might try different scaling variations later (start with CLIP ViT-B/32 and test larger ones later)
- Later will attempt object detection → Scaling/cropping image before scanning
