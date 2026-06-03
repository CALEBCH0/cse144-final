# CSE 144 Final Report

## 1 Introduction

1. **Problem goal and setting.** _TODO_
2. **Why transfer learning is appropriate.** _TODO_
3. **Brief summary of approach and main result.** Three pretrained CNNs (ResNet-50, ResNet-101, ConvNeXt-Base) were fine-tuned on a 100-class image classification task. ConvNeXt-Base achieved the best validation accuracy of 67.90% in the initial baseline run.

---

## 2 Dataset

1. **Number of classes and dataset sizes.** 100 classes; 1079 total labeled images split into ~917 train / ~162 validation (85/15 split, seed 42). Test set: _TODO (Kaggle)_.
2. **Directory structure and label mapping.** ImageFolder format under `data/train/`; labels 0–99 assigned alphabetically by folder name via HuggingFace `datasets` `imagefolder` loader.
3. **Preprocessing and augmentation.**
   - Train: `RandomResizedCrop`, `RandomHorizontalFlip`, `ToTensor`, ImageNet normalization
   - Validation: `Resize`, `CenterCrop`, `ToTensor`, ImageNet normalization

---

## 3 Implementation

### 3.1 Model

Three pretrained backbones were compared:

| Model | HuggingFace ID | Pretrained On |
|---|---|---|
| ResNet-50 | `microsoft/resnet-50` | ImageNet-1K (1,000 classes) |
| ResNet-101 | `microsoft/resnet-101` | ImageNet-1K (1,000 classes) |
| ConvNeXt-Base | `facebook/convnext-base-224-22k-1k` | ImageNet-22K → fine-tuned on 1K |

1. **Pretrained backbone.** ResNet-50/101 chosen as standard baselines. ConvNeXt-Base chosen as a modern CNN architecture with stronger pretraining (22K-class ImageNet), expected to outperform ResNets on transfer tasks.
2. **Architecture changes.** The final classification head was replaced: original 1000-class linear layer reinitialized to 100 classes (`ignore_mismatched_sizes=True`). All other weights initialized from pretrained checkpoint.
3. **Fine-tuning strategy.** Full fine-tuning (no layers frozen). All parameters updated end-to-end from the pretrained initialization.

### 3.2 Training

1. **Loss function and optimizer.** Cross-entropy loss. AdamW optimizer (HuggingFace Trainer default).
2. **Hyperparameters.**

| Parameter | ResNet-50 | ResNet-101 | ConvNeXt-Base |
|---|---|---|---|
| Learning rate | 1e-4 | 1e-4 | 5e-5 |
| Batch size | 128 | 128 | 128 |
| Epochs | 10 | 10 | 10 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| LR scheduler | linear warmup (Trainer default) | linear warmup | linear warmup |

3. **Hardware/software.** NVIDIA RTX 5070 Ti (16 GB VRAM); PyTorch 2.12.0+cu128; mixed precision (fp16); 4 dataloader workers.

---

## 4 Experiments

1. **Baseline setup.** All three models trained from scratch with the hyperparameters above, full fine-tuning, no augmentation beyond standard resize/flip. Results reported below.
2. **Hyperparameter tuning plan and validation method.** _TODO_
3. **Ablations planned.** _TODO (augmentations, freezing strategy, LR, model size)_

---

## 5 Results

### 5.1 Baseline Run (Epoch 10)

| Model | Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| ResNet-50 | Train | 0.1287 | 0.1085 | 0.0710 | 0.0594 |
| ResNet-50 | Validation | 0.0679 | 0.0134 | 0.0337 | 0.0171 |
| ResNet-101 | Train | 0.2236 | 0.1953 | 0.1499 | 0.1219 |
| ResNet-101 | Validation | 0.1481 | 0.0774 | 0.1118 | 0.0812 |
| ConvNeXt-Base | Train | 0.8942 | 0.9141 | 0.8840 | 0.8878 |
| ConvNeXt-Base | Validation | **0.6790** | 0.6491 | 0.6467 | 0.6239 |

All metrics are macro-averaged across 100 classes.

### 5.2 Extended Run — ConvNeXt-Base vs ConvNeXt-Large (batch=128, lr=5e-5, 10 epochs, GPU)

| Model | Split | Accuracy | Precision | Recall | F1 | Train-Val Gap |
|---|---|---|---|---|---|---|
| ConvNeXt-Base | Train | 0.6925 | 0.6778 | 0.6402 | 0.6247 | |
| ConvNeXt-Base | Validation | 0.4568 | 0.3441 | 0.3819 | 0.3327 | 23 pts |
| ConvNeXt-Large | Train | 0.8353 | 0.8414 | 0.8057 | 0.8019 | |
| ConvNeXt-Large | Validation | **0.5617** | 0.4866 | 0.5244 | 0.4814 | 27 pts |

ConvNeXt-Large improves validation accuracy by ~9 points over Base but overfits more (27 vs 23 point train-val gap), consistent with larger capacity memorizing the small training set.

### 5.3 5-Fold Stratified CV — DINOv2 / ConvNeXt-Tiny / ConvNeXt-Base / ViT (batch=32, label_smoothing=0.05, 10 epochs)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| DINOv2-Base (frozen) | 0.4133 ± 0.0334 | 0.3189 ± 0.0415 | 0.3645 ± 0.0425 | 0.3196 ± 0.0398 |
| ConvNeXt-Tiny | 0.2410 ± 0.0229 | 0.1449 ± 0.0338 | 0.1860 ± 0.0191 | 0.1397 ± 0.0219 |
| ConvNeXt-Base | **0.7071 ± 0.0274** | 0.6849 ± 0.0298 | 0.6980 ± 0.0250 | 0.6702 ± 0.0278 |
| ViT-B/16 (timm) | **0.7081 ± 0.0231** | 0.7063 ± 0.0383 | 0.7010 ± 0.0267 | 0.6795 ± 0.0295 |

All metrics are macro-averaged across 100 classes. Mean ± std across 5 folds.

**Key observations:**
- ConvNeXt-Base and ViT-B/16 are statistically tied (~70.7–70.8% accuracy, overlapping std). Both benefit from ImageNet-22K/21K pretraining.
- DINOv2 with a fully frozen backbone achieves only 41.3% — strong for a linear probe but limited by not updating any backbone features. Partial unfreezing expected to improve this significantly.
- ConvNeXt-Tiny underperforms at 24.1%, likely because `facebook/convnext-tiny-224` was only pretrained on ImageNet-1K (vs 22K for Base), giving it weaker feature representations for transfer.
- Low std values (~2–3%) confirm the 5-fold CV estimates are stable.

2. **Kaggle public leaderboard score.** _TODO_
3. **Qualitative error analysis.** _TODO_

---

## 6 Discussion

1. **What worked best and why.** ConvNeXt-Base significantly outperformed both ResNets (67.90% vs 14.81% val accuracy). The primary driver is pretraining data: ConvNeXt was pretrained on ImageNet-22K (22,000 classes) then fine-tuned on 1K, giving it far richer feature representations for transfer. ResNet-50 and ResNet-101 show clear overfitting (train acc ~12–22% vs val ~7–15%), suggesting the backbone features do not transfer as effectively to this dataset.
2. **Failure cases, overfitting/underfitting observations.** _TODO_
3. **Limitations and next improvements.** _TODO_

---

## 7 Reproducibility

1. **Random seeds.** `SEED = 42` passed to `set_seed()` (HuggingFace), `train_test_split`, and `TrainingArguments`.
2. **Environment setup.**
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. **Training command.**
   ```
   python pipeline.py
   ```
