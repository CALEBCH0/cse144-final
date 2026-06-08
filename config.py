SELECTED_MODELS = [
    "siglip2_so400m_512",
]

GPU_MEMORY_FRACTION = 0.85

SEED = 42
TRAIN_DIR = __import__("os").environ.get("TRAIN_DIR", "data/train")
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
USE_MIXUP = False              # Search (cfg 35): Mixup hurts ConvNeXt −1.95%; disabled
USE_CUTMIX = False             # Search (cfg 36/37): CutMix/MixupCutMix hurt −6%; disabled
USE_PROGRESSIVE_UNFREEZE = False # DINOv2: fixed unfreeze=2 is optimal; prog-unfreeze does full FT at ep3 which hurts
USE_LLRD = True                # Both DINOv2 models benefit: base=0.85, large=0.75 (per-model override in MODELS)
USE_TTA = True                 # Test-time augmentation (5 passes)
USE_ENSEMBLE = True            # Average logits from all fold models
USE_COLOR_JITTER = False       # Hurts ConvNeXt (−1.76%); disabled
USE_RANDAUGMENT = False        # Best DINOv2 config (84.24%) used no augmentation
USE_RANDOM_ERASING = False     # Marginal in full stack; disabled until more data
USE_LORA = False               # LoRA adapters (dinov2 / vit only); overrides freeze strategy
USE_CLASS_ROUTING = True       # Hard class routing: log-prob z-score + column-select by best per-class val F1
USE_PSEUDO_LABEL = False       # Disabled for solo validation run
PSEUDO_LABEL_THRESHOLD = 0.97  # Min softmax confidence to accept a pseudo-label
USE_CLASS_WEIGHTS = False      # Hurts all models (SigLIP-Base -0.75%, DINOv2-Large -2.13%); disabled
USE_UPSAMPLE_BALANCE = False   # Disabled for solo validation run
UPSAMPLE_MIN_COUNT = 20        # Target minimum images per class (never downsamples majority classes)
LORA_R = 8                     # LoRA rank (4/8/16)
LLRD_FACTOR = 0.75
PROGRESSIVE_UNFREEZE_EPOCH = 3
TTA_AUGMENTS = 5
UNFREEZE_BLOCKS = 2            # Best for DINOv2 (84.24%); 4 for ConvNeXt (see per-model override)
