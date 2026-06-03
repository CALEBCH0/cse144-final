SELECTED_MODELS = [
    "dinov2",
    "convnext",
    "vit",
]

GPU_MEMORY_FRACTION = 0.85

SEED = 42
TRAIN_DIR = "data/train"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
N_FOLDS = 5

# ── Technique toggles (set to best config from search) ────────────────────────
# Search findings:
#   DINOv2 unfreeze=2 → 84.24%  (best overall)
#   ConvNeXt lr=3e-4, unfreeze=4 → 76.27%  (see models/__init__.py for per-model lr)
#   LLRD: no benefit across all decay factors (±0.05%)
#   ColorJitter: hurts ConvNeXt (−1.76%); disabled
#   RandAugment: small consistent gain (+0.93%); enabled
#   CutMix / Mixup combos: not yet fully evaluated (configs 35–37 pending)
USE_MIXUP = True               # Mixup data augmentation (alpha=0.2)
USE_CUTMIX = False             # CutMix — pending search results (configs 35–37)
USE_PROGRESSIVE_UNFREEZE = True  # Unfreeze full backbone after epoch 3
USE_LLRD = False               # LLRD showed no benefit in search; disabled
USE_TTA = True                 # Test-time augmentation (5 passes)
USE_ENSEMBLE = True            # Average logits from all fold models
USE_COLOR_JITTER = False       # Hurts ConvNeXt (−1.76%); disabled
USE_RANDAUGMENT = True         # +0.93% on ConvNeXt; enabled
USE_RANDOM_ERASING = False     # Marginal in full stack; disabled until more data
USE_LORA = False               # LoRA adapters (dinov2 / vit only); overrides freeze strategy
LORA_R = 8                     # LoRA rank (4/8/16)
LLRD_FACTOR = 0.75
PROGRESSIVE_UNFREEZE_EPOCH = 3
TTA_AUGMENTS = 5
UNFREEZE_BLOCKS = 2            # Best for DINOv2 (84.24%); 4 for ConvNeXt (see per-model override)
