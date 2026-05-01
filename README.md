# Multimodal Fake News Detection

6-way fake news classification using BERT (text) + ResNet50 (image) with late fusion.

## Results
| Model | Test Accuracy | Macro-F1 |
|---|---|---|
| BERT (text only) | 72.82% | 0.67 |
| ResNet50 (image only) | 67.80% | 0.59 |
| Fusion (FeatureGatedFusion) | 75.43% | 0.63 |

## Dataset
[Fakeddit](https://github.com/entitize/Fakeddit) — 265K+ Reddit posts, 6-class labels

## Pipeline
1. `text_model_kaggle.ipynb` — Fine-tune BERT, export embeddings
2. `image_model_kaggle.ipynb` — Fine-tune ResNet50, export embeddings  
3. `fusion_model_kaggle.ipynb` — Train FeatureGatedFusion MLP
4. `inference_kaggle.ipynb` — Predict on new data

## Requirements
```bash
pip install torch torchvision transformers scikit-learn pandas seaborn tqdm
```

## Checkpoints
Download trained weights from [Releases](../../releases).
