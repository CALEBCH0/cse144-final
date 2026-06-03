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
3. **Fine-tuning strategy.** Partial fine-tuning with selective layer unfreezing. By default, the top 4 backbone blocks/stages are unfrozen alongside the classifier head. Five additional techniques are applied:

   | Technique | Description | Implementation |
   |---|---|---|
   | Layer-wise LR Decay (LLRD) | Earlier layers receive a lower LR multiplied by `decay_factor` per layer; classifier gets full `base_lr` | `get_llrd_optimizer()` in `utils.py` |
   | Mixup | Pairs of images blended with Beta(0.2, 0.2) lambda; cross-entropy on soft one-hot labels | `MixupCollator` + `CustomTrainer` in `utils.py` |
   | Progressive Unfreezing | Backbone frozen for first 3 epochs (head stabilizes), then fully unfrozen | `ProgressiveUnfreezeCallback` in `utils.py` |
   | Test-Time Augmentation (TTA) | Inference run 5× with random train augmentations; softmax probs averaged | `predict_with_tta()` in `utils.py` |
   | Ensemble | Logits from all 5 fold models averaged; final prediction from argmax | `ensemble_predict()` in `utils.py` |

   **Techniques considered but not implemented (TODO):**
   - Stochastic Weight Averaging (SWA) / Exponential Moving Average (EMA)
   - Focal loss for hard-class emphasis
   - CutMix / RandAugment / AutoAugment
   - Knowledge distillation (large → small model)
   - Larger backbone variants (SwinV2-Base, EfficientNetV2-M, DeiT III)

### 3.2 Training

1. **Loss function and optimizer.** Cross-entropy loss with label smoothing (0.05). AdamW optimizer with LLRD: classifier head uses `base_lr`, each encoder layer below is multiplied by `decay_factor=0.75`.
2. **Hyperparameters (CV run).**

| Parameter | ConvNeXt-Base | ViT-B/16 | DINOv2-Base |
|---|---|---|---|
| Learning rate (base) | 5e-5 | 5e-5 | 5e-5 |
| Batch size | 32 | 32 | 32 |
| Epochs | 10 | 10 | 10 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| LR scheduler | linear | linear | linear |
| Unfreeze blocks | 4 | 4 | 4 |
| LLRD decay factor | 0.75 | 0.75 | 0.75 |
| Mixup alpha | 0.2 | 0.2 | 0.2 |
| Progressive unfreeze epoch | 3 | 3 | 3 |
| TTA augments | 5 | 5 | 5 |

3. **Hardware/software.** NVIDIA RTX 5070 Ti (16 GB VRAM); PyTorch 2.12.0+cu128; mixed precision (fp16); `dataloader_num_workers=0` (Windows WSL2 multiprocessing constraint).

---

## 4 Experiments

1. **Baseline setup.** All three models trained with standard resize/flip augmentation, 5-fold stratified CV, AdamW, label smoothing 0.05. Results reported in sections 5.1–5.4.
2. **Hyperparameter tuning method.** Grid search via `search.py`: each config trained with 5-fold stratified CV (10 epochs, batch=32). Results written incrementally to `search_results.csv` with resume support. 44 total configs covering unfreeze depth, LR, scheduler, label smoothing, weight decay, LLRD, augmentation stack, collator (Mixup/CutMix), and LoRA rank. See section 5.5 for findings.
3. **Ablation axes covered:**
   - Freeze depth: head-only → partial unfreeze (1–4 blocks) → full fine-tune
   - LR: 1e-5 to 3e-4 per model family
   - Scheduler: linear vs cosine
   - LLRD: decay factors 0.65, 0.75, 0.85
   - Augmentation: ColorJitter, RandAugment, RandomErasing (isolated and combined)
   - Collator: none, Mixup, CutMix, MixupCutMix
   - PEFT: LoRA rank 4, 8, 16 on DINOv2 and ViT

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

### 5.4 5-Fold CV with All Techniques — DINOv2 / ConvNeXt-Base / ConvNeXt-Large / ViT (batch=32, 10 epochs, LLRD + Mixup + Progressive Unfreeze + TTA + Ensemble)

**Techniques active:** LLRD (decay=0.75), Mixup (α=0.2), Progressive Unfreeze (epoch 3), TTA (5 passes), Ensemble (5-fold average). Top 4 backbone blocks unfrozen initially.

| Model | Accuracy | Precision | Recall | F1 | Classes F1>0 |
|---|---|---|---|---|---|
| DINOv2-Base | 0.4133 ± 0.0334 | 0.3189 ± 0.0415 | 0.3645 ± 0.0425 | 0.3196 ± 0.0398 | 84/100 |
| ConvNeXt-Base | 0.7071 ± 0.0274 | 0.6849 ± 0.0298 | 0.6980 ± 0.0250 | 0.6702 ± 0.0278 | 98/100 |
| ConvNeXt-Large | **0.7785 ± 0.0302** | **0.7667 ± 0.0415** | **0.7756 ± 0.0284** | **0.7499 ± 0.0340** | 99/100 |
| ViT-B/16 | 0.6942 ± 0.0247 | 0.6800 ± 0.0312 | 0.6877 ± 0.0300 | 0.6634 ± 0.0283 | 98/100 |

All metrics macro-averaged across 100 classes. Mean ± std across 5 folds.

**Key observations:**
- ConvNeXt-Large is the clear winner at **77.85%** — a +7.1 point gain over Base (70.71%) and +13.5 points over the earlier single-split ConvNeXt-Large baseline (section 5.2). The selective unfreezing + progressive unfreeze strategy substantially reduces the overfitting seen in the full fine-tune baseline.
- DINOv2 unchanged at 41.3% — the 4-block partial unfreeze with progressive unfreeze did not noticeably improve over the fully frozen linear probe result. DINOv2 likely requires more epochs or higher LR once unfrozen.
- ViT-B/16 dropped slightly vs section 5.3 (0.6942 vs 0.7081), suggesting Mixup and/or LLRD may slightly hurt ViT at this dataset size. Further ablation needed.
- Class 66 scores F1=0.0000 across all four models — a genuinely hard class. Classes 64, 68, 71, 76, 86, 98 also score below 0.15 for all models.
- ConvNeXt-Large achieves F1=1.0 on 39/100 classes, vs 33/100 for Base — the larger pretrained capacity provides better feature coverage.

**Hard classes (F1 < 0.30 across all models):**

| Class | DINOv2 | ConvNeXt-B | ConvNeXt-L | ViT |
|---|---|---|---|---|
| 66 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 64 | 0.0000 | 0.2000 | 0.1333 | 0.0500 |
| 76 | 0.0000 | 0.1333 | 0.2667 | 0.0000 |
| 86 | 0.0000 | 0.0000 | 0.4267 | 0.3800 |
| 71 | 0.1467 | 0.2800 | 0.2286 | 0.1143 |

### 5.5 Hyperparameter Search — Key Findings (`search_results.csv`, 31/44 configs complete)

#### 5.5.1 DINOv2 unfreeze depth (biggest lever)

| unfreeze_blocks | acc | f1 |
|---|---|---|
| 0 (head only) | 0.4105 ± 0.0364 | 0.3175 |
| **2** | **0.8424 ± 0.0172** | **0.8194** |
| 4 | 0.8257 ± 0.0183 | 0.8043 |
| 12 (full FT, lr=1e-5) | 0.7739 ± 0.0155 | 0.7401 |

**Best overall config in the entire search.** Unfreezing just 2 top encoder layers achieves 84.24% — exceeding ConvNeXt-Large (77.85%) by +6.4 points. Unfreezing more layers _hurts_: 4 blocks −1.7 pts, full fine-tune −6.8 pts. Classic catastrophic forgetting on ~1,000 images; only the top two DINOv2 layers need adapting.

#### 5.5.2 ConvNeXt-Base LR sweep (unfreeze=4 stages)

| lr | acc |
|---|---|
| 1e-5 | 0.3160 |
| 3e-5 | 0.6497 |
| 5e-5 | 0.7146 |
| 1e-4 | 0.7544 |
| **3e-4** | **0.7627** |

ConvNeXt requires 10× higher LR than DINOv2. The frozen backbone + lr=1e-3 (head only) also performs surprisingly well at 73.87%, confirming that for CNN backbones, a high-LR head-only strategy can compete with partial fine-tuning.

#### 5.5.3 LLRD — no meaningful benefit

All LLRD decay factors (0.65 / 0.75 / 0.85) on ConvNeXt cluster within ±0.05% of the no-LLRD baseline (71.73%). LLRD is disabled in the final pipeline.

#### 5.5.4 Augmentation ablation (ConvNeXt, unfreeze=4, cosine)

| Augmentation | acc | Δ vs baseline |
|---|---|---|
| None (baseline) | 0.7173 | — |
| **+RandAugment only** | **0.7266** | **+0.93%** |
| +ColorJitter + RA + RandomErasing (full) | 0.7174 | +0.01% |
| +ColorJitter only | 0.6997 | −1.76% |

RandAugment alone gives the best gain. ColorJitter alone hurts — likely erasing subtle color discriminative cues the ConvNeXt backbone relies on. The full stack cancels out RandAugment's benefit when combined with ColorJitter.

#### 5.5.5 LoRA (frozen backbone)

| Model | rank r | acc | vs head-only |
|---|---|---|---|
| DINOv2 LoRA | 4 | 0.6562 | +24.6 pts |
| DINOv2 LoRA | 8 | 0.6692 | +25.9 pts |
| DINOv2 LoRA | **16** | **0.6719** | **+26.1 pts** |
| ViT LoRA | 4 | 0.5496 | — |
| ViT LoRA | 8 | 0.5515 | — |
| ViT LoRA | 16 | 0.5505 | — |

LoRA dramatically beats frozen head-only DINOv2 (41% → 67%) but cannot match partial unfreezing (84%). LoRA rank makes almost no difference — r=4 and r=16 are within 0.16%. ViT LoRA reaches ~55%, useful for parameter-efficient deployment.

#### 5.5.6 Still pending (configs 12–15, 18–19, 24, 28–30, 35–37)

ViT LR sweep (linear), ViT scheduler (cosine), ViT weight decay, ViT LLRD, and augmentation + Mixup/CutMix/MixupCutMix combos have not yet run. See `CONFIGS_TO_RUN` in `search.py`.

2. **Kaggle public leaderboard score.** _TODO_
3. **Qualitative error analysis.** _TODO_

---

## 6 Discussion

1. **What worked best and why.** ConvNeXt-Large with selective partial unfreezing (top 4 stages) and progressive unfreeze achieved 77.85% 5-fold CV accuracy — the best result across all experiments. The combination of stronger pretraining (ImageNet-22K), larger model capacity, and techniques to prevent premature overfitting (LLRD, Mixup, frozen warmup) resolved the 27-point train-val gap observed in the initial full fine-tune run. ConvNeXt-Base and ViT-B/16 are closely matched (~70%), both benefiting from 22K-class pretraining.
2. **Failure cases, overfitting/underfitting observations.** Class 66 scores F1=0 across all models, indicating the model consistently fails to identify it — possibly due to high visual similarity with neighboring classes or very few distinctive training examples. DINOv2 at 41.3% suggests its self-supervised features, while strong, may not align as well with the fine-grained visual differences in this dataset compared to supervised ImageNet pretraining.
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
