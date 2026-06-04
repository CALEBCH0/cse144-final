# CSE 144 Final Report

## 1 Introduction

1. **Problem goal and setting.** _TODO_
2. **Why transfer learning is appropriate.** _TODO_
3. **Brief summary of approach and main result.** Five pretrained backbones (ResNet-50, ResNet-101, ConvNeXt-Base, ViT-B/16, DINOv2-Base, DINOv2-Large) were evaluated on a 100-class image classification task. DINOv2-Large with 2 unfrozen encoder layers, lr=1e-4, LLRD=0.75, and 30 epochs achieved the best 5-fold CV accuracy of **86.84%**.

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

Six pretrained backbones were evaluated across the experiment:

| Model | HuggingFace ID | Pretrained On |
|---|---|---|
| ResNet-50 | `microsoft/resnet-50` | ImageNet-1K (1,000 classes) |
| ResNet-101 | `microsoft/resnet-101` | ImageNet-1K (1,000 classes) |
| ConvNeXt-Base | `facebook/convnext-base-224-22k-1k` | ImageNet-22K → fine-tuned on 1K |
| ViT-B/16 | `google/vit-base-patch16-224` | ImageNet-21K → fine-tuned on 1K |
| DINOv2-Base | `facebook/dinov2-base` | Self-supervised DINO (unlabeled ImageNet); 12 encoder layers |
| DINOv2-Large | `facebook/dinov2-large` | Self-supervised DINO (unlabeled ImageNet); 24 encoder layers |

1. **Pretrained backbone.** ResNet-50/101 chosen as standard baselines. ConvNeXt-Base chosen as a modern CNN with 22K pretraining. DINOv2's self-supervised pretraining produces highly transferable features that proved most effective for this small dataset.
2. **Architecture changes.** The final classification head was replaced: original 1000-class linear layer reinitialized to 100 classes (`ignore_mismatched_sizes=True`). All other weights initialized from pretrained checkpoint.
3. **Fine-tuning strategy.** Partial fine-tuning with selective layer unfreezing. By default, the top 4 backbone blocks/stages are unfrozen alongside the classifier head. Five additional techniques are applied:

   | Technique | Description | Implementation |
   |---|---|---|
   | Layer-wise LR Decay (LLRD) | Earlier layers receive a lower LR multiplied by `decay_factor` per layer; classifier gets full `base_lr` | `get_llrd_optimizer()` in `utils.py` |
   | Mixup | Pairs of images blended with Beta(0.2, 0.2) lambda; cross-entropy on soft one-hot labels | `MixupCollator` + `CustomTrainer` in `utils.py` |
   | Progressive Unfreezing | Backbone frozen for first 3 epochs (head stabilizes), then fully unfrozen | `ProgressiveUnfreezeCallback` in `utils.py` |
   | Test-Time Augmentation (TTA) | Inference run 5× with random train augmentations; softmax probs averaged | `predict_with_tta()` in `utils.py` |
   | Ensemble | Logits from all 5 fold models averaged; final prediction from argmax | `ensemble_predict()` in `utils.py` |

   **Techniques considered but not implemented:**
   - Stochastic Weight Averaging (SWA) / Exponential Moving Average (EMA)
   - Focal loss for hard-class emphasis
   - Knowledge distillation (large → small model)
   - Larger backbone variants (SwinV2-Base, EfficientNetV2-M, DeiT III)

### 3.2 Training

1. **Loss function and optimizer.** Cross-entropy loss with label smoothing (0.05). AdamW optimizer with LLRD: classifier head uses `base_lr`, each encoder layer below is multiplied by `decay_factor=0.75`.
2. **Hyperparameters (CV run).**

| Parameter | ConvNeXt-Base | ViT-B/16 | DINOv2-Base | DINOv2-Large |
|---|---|---|---|---|
| Learning rate (base) | **3e-4** | **1e-4** | **1e-4** | **1e-4** |
| Batch size | 32 | 32 | 32 | 32 |
| Epochs | 10 | 10 | **10** | **30** |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| LR scheduler | linear | cosine | linear | linear |
| Unfreeze blocks | 4 | **4** | **2** | **2** |
| LLRD decay factor | — (disabled) | **0.85** | **0.85** | **0.75** |
| Mixup alpha | — (disabled) | — | — | — |
| RandAugment | — | — | — | — |
| Progressive unfreeze | — | — | **OFF** | **OFF** |
| TTA augments | 5 | 5 | 5 | 5 |

Bold values reflect the best configuration found via hyperparameter search. DINOv2-base peaks at 10 epochs; DINOv2-Large benefits significantly from 30 epochs. LLRD is beneficial for ViT (0.85), DINOv2-base (0.85), and DINOv2-Large (0.75) but neutral for ConvNeXt. Mixup, CutMix, and RandAugment disabled for DINOv2 — aug degrades already-robust DINO features. Progressive unfreeze disabled for DINOv2 — see section 5.5.9.

3. **Hardware/software.** NVIDIA RTX 5070 Ti (16 GB VRAM); PyTorch 2.12.0+cu128; mixed precision (fp16); `dataloader_num_workers=0` (Windows WSL2 multiprocessing constraint). Scripts are run via Windows Python (`.venv-win/Scripts/python.exe`) from WSL2 to avoid the `/mnt/c/` filesystem bridge overhead for native CUDA throughput.

---

## 4 Experiments

1. **Baseline setup.** All three models trained with standard resize/flip augmentation, 5-fold stratified CV, AdamW, label smoothing 0.05. Results reported in sections 5.1–5.4.
2. **Hyperparameter tuning method.** Grid search via `search.py`: each config trained with 5-fold stratified CV (batch=32). Results written incrementally to `search_results.csv` with resume support. 78 total configs covering unfreeze depth, LR, scheduler, label smoothing, weight decay, LLRD, augmentation stack, collator (Mixup/CutMix), LoRA rank, epoch count, and two DINOv2 model sizes. Each config also generates a Kaggle-submittable CSV in `search_results/config_{i}_{model}.csv`. See section 5.5 for findings.
3. **Ablation axes covered:**
   - Freeze depth: head-only → partial unfreeze (1–6 blocks) → full fine-tune
   - LR: 1e-5 to 3e-4 per model family
   - Scheduler: linear vs cosine
   - LLRD: decay factors 0.65, 0.75, 0.85
   - Augmentation: ColorJitter, RandAugment, RandomErasing (isolated and combined)
   - Collator: none, Mixup, CutMix, MixupCutMix
   - PEFT: LoRA rank 4, 8, 16 on DINOv2 and ViT
   - Epochs: 10, 20, 30 (DINOv2-Base and DINOv2-Large)
   - Model scale: DINOv2-Base (12 layers) vs DINOv2-Large (24 layers)

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

### 5.5 Hyperparameter Search — Key Findings (`search_results.csv`, 78/78 configs complete)

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

#### 5.5.5 ViT-B/16 LR sweep (frozen backbone, unfreeze=0)

| Config | LR | Scheduler | Weight Decay | Acc |
|---|---|---|---|---|
| 12 | 1e-5 | linear | 0.01 | 0.0232 |
| 13 | 3e-5 | linear | 0.01 | 0.1010 |
| 14 | 5e-5 | linear | 0.01 | 0.2290 |
| **15** | **1e-4** | **linear** | **0.01** | **0.4495** |
| 18 | 5e-5 | cosine | 0.01 | 0.2392 |
| **19** | **1e-4** | **cosine** | **0.01** | **0.4523** |
| 24 | 5e-5 | linear | 0.05 | 0.2290 |

With a frozen backbone (head-only training), ViT-B/16 tops out at **45.2%** — dramatically below DINOv2 (84.2%) under comparable conditions. The LR sensitivity is steep: lr=1e-5 collapses to 2.3% while lr=1e-4 reaches 45%. Cosine scheduler matches linear at the same LR (45.23% vs 44.95%). Higher weight decay (0.05) offers no improvement over the default (0.01). Backbone unfreezing (configs 28–30) closes this gap substantially — see section 5.5.7.

#### 5.5.6 LoRA (frozen backbone)

| Model | rank r | acc | vs head-only |
|---|---|---|---|
| DINOv2 LoRA | 4 | 0.6562 | +24.6 pts |
| DINOv2 LoRA | 8 | 0.6692 | +25.9 pts |
| DINOv2 LoRA | **16** | **0.6719** | **+26.1 pts** |
| ViT LoRA | 4 | 0.5496 | +4.7 pts vs cfg 15 |
| ViT LoRA | 8 | 0.5515 | +4.9 pts |
| ViT LoRA | 16 | 0.5505 | +4.8 pts |

LoRA dramatically beats frozen head-only DINOv2 (41% → 67%) but cannot match partial unfreezing (84%). LoRA rank makes almost no difference — r=4 and r=16 are within 0.16%. For ViT, LoRA at ~55% modestly outperforms the best frozen-backbone baseline (45%), but still leaves ViT well below DINOv2. LoRA is useful for parameter-efficient deployment but not accuracy maximization at this dataset scale.

#### 5.5.7 ViT-B/16 with partial unfreezing + LLRD (configs 28–30)

| Config | unfreeze | LLRD factor | Acc | F1 | train/fold |
|---|---|---|---|---|---|
| 28 | 4 | 0.65 | 0.7146 | 0.6875 | 45s |
| 29 | 4 | 0.75 | 0.7183 | 0.6903 | 45s |
| **30** | **4** | **0.85** | **0.7257** | **0.6974** | **47s** |

Unfreezing 4 ViT encoder blocks with LLRD pushes accuracy from 45% (frozen) to **72.6%** — a +27 point jump. The gentlest decay (0.85) works best, suggesting ViT's pretrained features need only light adjustment. At 72.6%, ViT is competitive with ConvNeXt-Base (76.3%) and within reach of ConvNeXt-Large (77.9%), though still 12 points below DINOv2 (84.2%).

**Note:** unlike ConvNeXt, LLRD is beneficial for ViT — without it the frozen ViT reaches only 45%. The difference is that LLRD enables backbone layers to update while the frozen run trains the head alone.

#### 5.5.8 Collator ablation (ConvNeXt, full aug stack + Mixup / CutMix / MixupCutMix)

These configs use ConvNeXt-Base with the full augmentation stack (CJ + RA + RE) and add a label-mixing collator on top. All three **hurt** accuracy relative to no collator (config 34: 71.74%):

| Config | Collator | Acc | F1 | Δ vs no-collator |
|---|---|---|---|---|
| 34 | none | 0.7174 | 0.6798 | — |
| 35 | Mixup (α=0.2) | 0.6979 | 0.6555 | **−1.95%** |
| 36 | CutMix (α=1.0) | 0.6544 | 0.5981 | **−6.30%** |
| 37 | MixupCutMix | 0.6608 | 0.6097 | **−5.66%** |

All three collators consistently hurt. With ~10 images/class and 10 training epochs, soft labels from label blending add noise faster than they regularize. CutMix's large rectangular patches are particularly disruptive at this dataset scale. **Both Mixup and CutMix are disabled in the final pipeline.**

*(These configs previously failed due to a bug where `MixupCollator` applied label blending to validation images during prediction — HuggingFace Trainer iterates the DataLoader outside `torch.no_grad()`, so the grad-enabled check did not fire. Fix: swap to a plain collator before `.predict()`.)*

#### 5.5.9 Progressive unfreeze incompatibility with DINOv2

Running `pipeline.py` with `USE_PROGRESSIVE_UNFREEZE=True` and `UNFREEZE_BLOCKS=2` produced only **63.02%** — far below the 84.24% search result. The cause: `ProgressiveUnfreezeCallback` unfreezes the **entire model** at epoch 3 (not just the top-N blocks), effectively switching to full fine-tuning at `lr=5e-5` for the remaining 7 epochs. Since the search showed full fine-tuning (unfreeze=12, `lr=1e-5`) only reaches 77.4% — and at `lr=5e-5` it collapses further — the progressive unfreeze was actively destroying the learned representations.

**Fix:** `USE_PROGRESSIVE_UNFREEZE = False` for DINOv2. The 84.24% result used a fixed unfreeze=2 for all 10 epochs with no progressive strategy.

Progressive unfreeze is appropriate for ConvNeXt (where it stabilizes the head before full fine-tuning) but is incompatible with the DINOv2 strategy of keeping the backbone mostly frozen throughout training.

#### 5.5.10 DINOv2-Base LR sweep at unfreeze=2 (configs 44–47)

| lr | acc | notes |
|---|---|---|
| 1e-5 | 0.7489 | too conservative; features don't adapt |
| 3e-5 | 0.8313 | |
| **5e-5** | **0.8424** | original best (config 1) |
| 1e-4 | 0.8359 | slight drop vs 5e-5 at 10ep alone |
| 3e-4 | 0.8007 | too aggressive; catastrophic forgetting |

5e-5 is the optimal standalone LR. However, lr=1e-4 combined with LLRD (section 5.5.15) ultimately exceeds 5e-5 — LLRD shields the lower layers from the higher LR.

#### 5.5.11 DINOv2-Base LLRD sweep (configs 48–50)

| LLRD factor | acc | delta vs no LLRD (0.8424) |
|---|---|---|
| 0.65 | 0.8415 | −0.09% |
| 0.75 | 0.8425 | +0.01% |
| **0.85** | **0.8434** | **+0.10%** |

Unlike ConvNeXt (where LLRD was neutral), DINOv2-base shows a small but consistent benefit. Gentle decay (0.85) preserves more of the top-layer gradient while still protecting lower representations. The payoff is modest at 5e-5; it becomes significant when combined with a higher base LR (see section 5.5.15).

#### 5.5.12 DINOv2-Base augmentation ablation (configs 51–52)

| augmentation | acc | delta vs baseline (0.8424) |
|---|---|---|
| +RandAugment only | 0.8406 | −0.18% |
| +CJ + RA + RE (full) | 0.8322 | −1.02% |

Both hurt. DINOv2's DINO-pretrained features are already highly invariant; additional augmentation degrades fine-grained discriminative signals without providing useful generalisation variance at ~10 images/class. Compare to ConvNeXt where RA alone gave +0.93%.

#### 5.5.13 ViT-B/16 augmentation ablation (configs 53–55)

Best ViT base: unfreeze=4, LLRD=0.85, lr=5e-5, cosine (0.7257, config 30).

| augmentation | acc | delta |
|---|---|---|
| baseline | 0.7257 | — |
| +RandAugment only | 0.7331 | +0.74% |
| **+ColorJitter only** | **0.7442** | **+1.85%** |
| +CJ + RA + RE (full) | 0.7257 | 0.00% |

ColorJitter alone is the best ViT augmentation — a different pattern from ConvNeXt (where CJ hurt −1.76%). ViT's patch attention mechanism benefits from color variation; ConvNeXt's deep feature hierarchy is more sensitive to color shift. The full aug stack cancels out gains, consistent with the ConvNeXt finding.

**Updated best ViT:** 74.42% (config 54: unfreeze=4, LLRD=0.85, +ColorJitter).

#### 5.5.14 DINOv2-Large — introduction and initial results (configs 56–63)

DINOv2-Large (`facebook/dinov2-large`) uses 24 encoder layers vs 12 for Base. A layer-count bug in `apply_freeze` (hardcoded to 12) caused configs 57–58 to unfreeze layers 10–11 (middle of the 24-layer network) instead of 22–23 (top two). Configs 57–58 are invalid (acc ~0.52). Fixed by adding `_count_transformer_layers()` which scans parameter names dynamically.

Post-fix results (10-epoch baseline):

| config | unfreeze | lr | epochs | aug | acc |
|---|---|---|---|---|---|
| 56 | 0 (linear probe) | 5e-5 | 10 | — | 0.4986 |
| 59 | 2 | 1e-5 | 10 | — | 0.7803 |
| 60 | 2 | 1e-4 | 10 | — | 0.8517 |
| 61 | 2 | 5e-5 | 20 | — | 0.8545 |
| **62** | **2** | **5e-5** | **30** | — | **0.8665** |
| 63 | 2 | 5e-5 | 10 | RandAugment | 0.8573 |

**Key finding:** unlike DINOv2-base (which peaks at 10 epochs), Large benefits substantially from extended training. 30ep → 0.8665 vs 10ep (config 61 20ep → 0.8545 → config 60 10ep → 0.8517). Each additional epoch block adds ~1 point. DINOv2-Large has more capacity to absorb the small dataset over more passes without the catastrophic forgetting seen with full fine-tuning.

#### 5.5.15 DINOv2-Base best combos — lr=1e-4 + LLRD + depth (configs 64–69)

Combining the best LR (1e-4) with the best LLRD factor (0.85) and probing deeper unfreeze + longer training:

| config | unfreeze | lr | LLRD | epochs | acc |
|---|---|---|---|---|---|
| **64** | 2 | **1e-4** | **0.85** | 10 | **0.8554** |
| 65 | 2 | 1e-4 | 0.75 | 10 | 0.8517 |
| 66 | 4 | 1e-4 | 0.85 | 10 | 0.8146 |
| 67 | 4 | 5e-5 | 0.85 | 10 | 0.8387 |
| 68 | 6 | 5e-5 | 0.85 | 10 | 0.8146 |
| 69 | 2 | 1e-4 | 0.85 | **20** | 0.8526 |

**New DINOv2-base best: 85.54%** — combining lr=1e-4 with LLRD=0.85 exceeds the prior best (5e-5 no LLRD) by +1.3 points. The LLRD allows a higher head/top-layer LR while keeping lower layers at 1e-4 × 0.85^n. Unfreeze=4 and =6 still hurt significantly, confirming that 2 top blocks is the correct freeze depth regardless of LLRD. Extending to 20ep regresses (0.8526 < 0.8554) — DINOv2-base overfits past 10 epochs on this dataset.

#### 5.5.16 DINOv2-Large overnight — best configs (configs 70–77)

Combining all best findings (lr=1e-4, 30ep, LLRD, selective augmentation) for Large:

| config | lr | LLRD | epochs | aug | acc |
|---|---|---|---|---|---|
| 70 | 1e-4 | 0.85 | 30 | — | 0.8665 |
| 71 | 1e-4 | 0.85 | 20 | — | 0.8564 |
| **72** | **1e-4** | **0.75** | **30** | — | **0.8684** |
| 73 | 1e-4 | 0.85 | 30 | RandAugment | 0.8638 |
| 74 | 1e-4 | none | 30 | — | 0.8619 |
| 75 | 5e-5 | 0.85 | 30 | — | 0.8563 |
| 76 (base) | 1e-4 | 0.85 | 20 | — | 0.8499 |
| 77 (base) | 1e-4 | 0.85 | 30 | — | 0.8406 |

**New overall best: 86.84%** (config 72: DINOv2-Large, unfreeze=2, lr=1e-4, LLRD=0.75, 30ep).

Key takeaways:
- **LLRD matters for Large at lr=1e-4:** LLRD=0.75 → 0.8684 vs no LLRD → 0.8619 (+0.65 pts); LLRD=0.85 → 0.8665. Gentler decay (0.75) performs best, allowing the top layers more gradient while protecting lower representations from a high LR.
- **30 epochs is necessary:** 20ep → 0.8564, 30ep → 0.8665 (same LLRD=0.85, lr=1e-4). Large consistently absorbs more training without overfitting.
- **RandAugment still slightly hurts:** 0.8638 vs 0.8665 at same settings — consistent with the DINOv2-base augmentation finding.
- **DINOv2-base with 20/30ep regresses:** 76 (20ep) → 0.8499, 77 (30ep) → 0.8406 vs best base 10ep → 0.8554. Base overfits past 10 epochs; Large does not.
- **lr=5e-5 vs 1e-4 for Large:** at 30ep, 5e-5 + LLRD=0.85 → 0.8563, 1e-4 + LLRD=0.75 → 0.8684. Higher LR requires LLRD to protect earlier layers but yields +1.2 points.

### 5.6 Overall Best Results Summary

| Model | Best acc | Config | Key settings |
|---|---|---|---|
| ResNet-50 | 6.79% | baseline | — |
| ResNet-101 | 14.81% | baseline | — |
| ConvNeXt-Base | 76.27% | 11 | unfreeze=4, lr=3e-4 |
| ConvNeXt-Large | 77.85% | pipeline | unfreeze=4, prog-unfreeze |
| ViT-B/16 | 74.42% | 54 | unfreeze=4, LLRD=0.85, +CJ |
| DINOv2-Base | 85.54% | 64 | unfreeze=2, lr=1e-4, LLRD=0.85, 10ep |
| **DINOv2-Large** | **86.84%** | **72** | **unfreeze=2, lr=1e-4, LLRD=0.75, 30ep** |

2. **Kaggle public leaderboard score.** _TODO_
3. **Qualitative error analysis.** _TODO_

---

## 6 Discussion

1. **What worked best and why.** DINOv2-Large with 2 unfrozen top encoder blocks, lr=1e-4, LLRD=0.75, and 30 epochs achieved **86.84%** — the best result across all 78 configs. DINOv2-Base peaked at **85.54%** (unfreeze=2, lr=1e-4, LLRD=0.85, 10ep). Both significantly exceed CNN and ViT baselines: ConvNeXt-Large 77.85%, ViT-B/16 74.42%. The pattern is consistent: DINOv2's self-supervised DINO pretraining yields features so transferable that only the top two encoder layers need adapting (unfreeze=4 loses −1.7 pts; unfreeze=6 loses −4.1 pts). Higher LR (1e-4) with LLRD outperforms the standalone optimal (5e-5 no LLRD) because LLRD lets the head and top layers update aggressively while shielding the pretrained lower representations. DINOv2-Large absorbs 30 epochs without overfitting due to its larger capacity; DINOv2-Base peaks at 10ep and regresses beyond that.

2. **Failure cases, overfitting/underfitting observations.** Class 66 scores F1=0 across all models — consistently the hardest class. DINOv2-base overfits past 10 epochs: 20ep → 0.8526 < 0.8554, 30ep → 0.8406 (using best config 64 params). DINOv2-Large does not show this: 30ep → 0.8684 > 20ep → 0.8564. ViT-B/16's steep LR sensitivity (2% frozen at 1e-5, 45% at 1e-4) reflects the CLS token struggling to adapt without backbone gradient flow; partial unfreezing with LLRD resolves it (+27 pts). The layer-count bug in `apply_freeze` (hardcoded 12 layers) made configs 57–58 invalid — DINOv2-Large has 24 layers, so unfreeze=2 was actually unfreezing layers 10–11 instead of 22–23.

3. **Limitations and next steps.**
   - **DINOv2-Large with TTA and ensemble not yet run:** The 86.84% search result uses no TTA or 5-fold ensemble averaging. Running `pipeline.py` with the config 72 settings and TTA is the immediate next step and likely adds 0.5–1 point.
   - **Per-class ensemble potential:** The resulter.py per-class F1 delta table enables class-routing — routing each test image to the model that historically dominates that class rather than averaging all models. Worth evaluating if Large and Base are complementary on specific classes.
   - **Label-mixing collators ineffective at this scale:** Mixup (−2%), CutMix (−6%), MixupCutMix (−6%) all hurt at ~10 images/class. Disabled in the final pipeline.
   - **Augmentation is neutral-to-harmful for DINOv2:** RandAugment and the full aug stack consistently degrade both Base and Large. Only ViT benefits from ColorJitter.
   - **Small dataset ceiling:** With ~10 images per class, further gains likely require semi-supervised pseudo-labeling on the test set or external data augmentation from similar public datasets.
   - **Kaggle submissions available per config:** `search_results/config_{i}_{model}.csv` provides a Kaggle-submittable prediction file for every completed config.

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
