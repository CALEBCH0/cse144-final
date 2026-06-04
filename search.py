import csv
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, set_seed

from config import GPU_MEMORY_FRACTION, N_FOLDS, SEED, TRAIN_DIR
from models import MODELS
from pipeline import collate_fn, compute_metrics
from transforms import build_transforms
from utils import (
    CutMixCollator,
    CustomTrainer,
    EpochAccuracyCallback,
    MixupCollator,
    MixupCutMixCollator,
    apply_freeze as _apply_freeze,
    apply_lora,
    predict_test_images,
    signal,
)

RESULTS_FILE = "search_results.csv"
SUBMISSIONS_DIR = "search_results"

# ── Configs to run ────────────────────────────────────────────────────────────
# Set to None to run everything not yet in the CSV (full resume mode).
# Set to a set of indices to run only those specific configs.
# Completed configs (already in search_results.csv) are always skipped regardless.
CONFIGS_TO_RUN = {
    # DINOv2 unfreeze=2 LR sweep
    # 44, 45, 46, 47,
    # DINOv2 unfreeze=2 LLRD sweep
    # 48, 49, 50,
    # DINOv2 unfreeze=2 augmentation ablation
    # 51, 52,
    # ViT unfreeze=4 augmentation ablation (best ViT: LLRD=0.85)
    # 53, 54, 55,
    # DINOv2-Large: unfreeze depth, LR sweep, epochs, augmentation
    # 56, 57, 58, 59, 60, 61, 62, 63,
    # DINOv2-base: best lr + best LLRD combos, more unfreeze, longer training
    # 64, 65, 66, 67, 68, 69,
    # DINOv2-Large + DINOv2-base: combine best findings, overnight run
    # 70, 71, 72, 73, 74, 75, 76, 77,
    # More epochs + LLRD sweep + cosine + deeper unfreeze for Large; base cosine
    # 78, 79, 80, 81, 82, 83, 84,  # done / currently running
    # 518px resolution + RandAugment combos
    85, 86, 87,
}

# fmt: off
SEARCH_CONFIGS = [
    # ── DINOv2 unfreeze depth ─────────────────────────────────────
    {"model": "dinov2", "unfreeze_blocks": 0,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 4,  "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 12, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ConvNeXt-Base freeze strategy ────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 1e-3,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 1, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ConvNeXt-Base LR sweep (full fine-tune) ───────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 3e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── ViT LR sweep ──────────────────────────────────────────────
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── LR scheduler ──────────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 1e-4,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── Label smoothing ───────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.0,  "weight_decay": 0.01},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.1,  "weight_decay": 0.01},

    # ── Weight decay ──────────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.05},
    {"model": "convnext", "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.1},
    {"model": "vit",      "unfreeze_blocks": 0, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.05},

    # ── LLRD decay factor ─────────────────────────────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "vit",      "unfreeze_blocks": 4, "lr": 5e-5,  "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},

    # ── Augmentation ablation (convnext, unfreeze=4, cosine) ──────
    # baseline: no extra aug, no mixup/cutmix
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": False, "use_randaugment": False, "use_random_erasing": False, "collator": "none"},
    # +color jitter only
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": False, "use_random_erasing": False, "collator": "none"},
    # +RandAugment only
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": False, "use_randaugment": True,  "use_random_erasing": False, "collator": "none"},
    # full aug stack (CJ + RA + RE)
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "none"},
    # full aug + mixup
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "mixup"},

    # ── CutMix vs Mixup (convnext, full aug stack) ────────────────
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "cutmix"},
    {"model": "convnext", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True,  "use_randaugment": True,  "use_random_erasing": True,  "collator": "mixup_cutmix"},

    # ── LoRA rank sweep — DINOv2 ──────────────────────────────────
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 4},
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 8},
    {"model": "dinov2", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 16},

    # ── LoRA rank sweep — ViT ─────────────────────────────────────
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 4},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 8},
    {"model": "vit", "unfreeze_blocks": 0, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "lora_r": 16},

    # ── DINOv2 LR sweep (unfreeze=2, best depth) ──────────────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 3e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 3e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2 LLRD (unfreeze=2, lr=5e-5, cosine) ────────────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.65},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.75},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85},

    # ── DINOv2 augmentation (unfreeze=2, lr=5e-5, linear) ─────────
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_randaugment": True},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_color_jitter": True, "use_randaugment": True, "use_random_erasing": True},

    # ── ViT augmentation (unfreeze=4, LLRD=0.85, best ViT config) ─
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_randaugment": True},
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_color_jitter": True},
    {"model": "vit", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01, "llrd_factor": 0.85,
     "use_color_jitter": True, "use_randaugment": True, "use_random_erasing": True},

    # ── DINOv2-Large: unfreeze depth (mirror best-of-Base findings) ─
    {"model": "dinov2_large", "unfreeze_blocks": 0,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 2,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 4,  "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2-Large: LR sweep at unfreeze=2 ──────────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01},

    # ── DINOv2-Large: more epochs at best unfreeze ─────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 20},
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 30},

    # ── DINOv2-Large: RandAugment at best unfreeze ────────────────
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5,  "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "use_randaugment": True},

    # ── DINOv2-base: combine best lr (1e-4) + best LLRD (0.85) ────  idx 64-69
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75},
    {"model": "dinov2", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 4, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 6, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85},
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},

    # ── DINOv2-Large: combine best findings for overnight run ──────  idx 70-77
    # A: best lr + best LLRD + best epochs (top priority)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},
    # B: same but 20ep — faster sanity check
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},
    # C: alt LLRD at full power
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # D: aug helped Large — combine with best settings
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30, "use_randaugment": True},
    # E: confirm LLRD effect at 30ep (no LLRD baseline)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "num_epochs": 30},
    # F: add LLRD to current best config (lr=5e-5, 30ep)
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 5e-5, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},
    # G: base best config + 20ep
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 20},
    # H: base best config + 30ep
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 30},

    # ── Next search: more epochs + LLRD sweep + cosine + deeper unfreeze ──  idx 78-84
    # 78: continue LLRD sweep — 0.85→0.75 gained +0.19%; does 0.65 continue the trend?
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 79: 40ep — Large still improving at 30ep (+1.2% over 20ep); expect +0.3-0.5% more
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 40},
    # 80: 50ep — queue alongside 40ep to avoid wasting a night if 40ep still gains
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 50},
    # 81: cosine at 30ep — long training commonly benefits from cosine LR decay
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30},
    # 82: unfreeze=4 for Large — 4/24 layers is only 17% (vs 33% for base); safe with LLRD=0.65
    {"model": "dinov2_large", "unfreeze_blocks": 4, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 83: push LR higher — needs even gentler LLRD to protect lower layers
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 2e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.65, "num_epochs": 30},
    # 84: base cosine — only untested axis for DINOv2-base
    {"model": "dinov2", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "cosine", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.85, "num_epochs": 10},

    # ── 518px resolution + RandAugment combos ─────────────────────────  idx 85-87
    # 85: dinov2_large at native 518px (patch_size=14 → 37×37 patches vs 16×16 at 224px)
    #     batch_size=16 to stay within VRAM (5× more tokens → higher activation memory)
    {"model": "dinov2_large_518", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "batch_size": 16},
    # 86: RandAugment on 224px Large — +0.93% on base but never tested on Large; isolate aug effect
    {"model": "dinov2_large", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "use_randaugment": True},
    # 87: 518px + RandAugment — combined; only run after 85/86 confirm both are individually positive
    {"model": "dinov2_large_518", "unfreeze_blocks": 2, "lr": 1e-4, "scheduler": "linear", "label_smoothing": 0.05, "weight_decay": 0.01,
     "llrd_factor": 0.75, "num_epochs": 30, "use_randaugment": True, "batch_size": 16},
]
# fmt: on

NUM_EPOCHS = 10
BATCH_SIZE = 32


def run_cv(cfg, model_cfg, full_dataset, fold_splits, class_names, num_labels, label2id, id2label, cfg_index=0):
    image_processor = model_cfg["get_processor"]()
    train_tf, val_tf = build_transforms(
        image_processor,
        use_color_jitter=cfg.get("use_color_jitter", False),
        use_randaugment=cfg.get("use_randaugment", False),
        use_random_erasing=cfg.get("use_random_erasing", False),
    )
    all_label_ids = list(range(num_labels))

    # Collator selection
    collator_type = cfg.get("collator", "none")
    if collator_type == "mixup":
        data_collator = MixupCollator(num_labels, alpha=0.2)
    elif collator_type == "cutmix":
        data_collator = CutMixCollator(num_labels, alpha=1.0)
    elif collator_type == "mixup_cutmix":
        data_collator = MixupCutMixCollator(num_labels, mixup_alpha=0.2, cutmix_alpha=1.0)
    else:
        data_collator = collate_fn

    fold_metrics = []
    fold_train_sec = []
    fold_eval_sec = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        signal("fold_start", config=cfg_index, fold=fold_idx + 1, n_folds=N_FOLDS, model=cfg["model"])
        train_fold = full_dataset.select(train_idx).with_transform(train_tf)
        val_fold   = full_dataset.select(val_idx).with_transform(val_tf)

        model = model_cfg["get_model"](num_labels, label2id, id2label)

        lora_r = cfg.get("lora_r", 0)
        if lora_r > 0:
            model = apply_lora(model, cfg["model"], r=lora_r)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            trainable = _apply_freeze(model, cfg["model"], cfg["unfreeze_blocks"])

        fold_batch_size = cfg.get("batch_size", BATCH_SIZE)
        training_args = TrainingArguments(
            output_dir=os.path.join(model_cfg["output_dir"], "search", f"fold_{fold_idx}"),
            num_train_epochs=cfg.get("num_epochs", NUM_EPOCHS),
            per_device_train_batch_size=fold_batch_size,
            per_device_eval_batch_size=fold_batch_size,
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            label_smoothing_factor=cfg["label_smoothing"],
            lr_scheduler_type=cfg["scheduler"],
            warmup_ratio=0.1 if cfg["scheduler"] == "cosine" else 0.0,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=SEED,
            remove_unused_columns=False,
            logging_strategy="no",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        # LLRD disabled when LoRA active (PEFT renames params, breaking layer pattern matching)
        llrd_factor = cfg.get("llrd_factor", 1.0) if lora_r == 0 else 1.0
        use_soft_labels = collator_type in ("mixup", "cutmix", "mixup_cutmix")
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            compute_metrics=compute_metrics,
            processing_class=image_processor,
            data_collator=data_collator,
            llrd_factor=llrd_factor,
            model_name=cfg["model"],
            use_mixup=use_soft_labels,
        )

        train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)
        trainer.add_callback(EpochAccuracyCallback(train_eval_fold, val_fold, collate_fn))

        try:
            t0 = time.perf_counter()
            trainer.train()
            fold_train_sec.append(time.perf_counter() - t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    fold {fold_idx+1}: OOM — skipping fold")
                torch.cuda.empty_cache()
                continue
            raise

        try:
            trainer.data_collator = collate_fn  # plain collator — no mixup on val images
            t0 = time.perf_counter()
            preds_out = trainer.predict(val_fold)
            fold_eval_sec.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"    fold {fold_idx+1}: predict failed ({e}) — skipping fold")
            torch.cuda.empty_cache()
            continue

        preds = np.argmax(preds_out.predictions, axis=1)
        true  = preds_out.label_ids

        fold_metrics.append({
            "accuracy":  accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro", zero_division=0),
            "recall":    recall_score(true, preds, average="macro", zero_division=0),
            "f1":        f1_score(true, preds, labels=all_label_ids, average="macro", zero_division=0),
        })
        fold_models.append(model.cpu())
        torch.cuda.empty_cache()
        tr = fold_train_sec[-1] if fold_train_sec else 0
        ev = fold_eval_sec[-1] if fold_eval_sec else 0
        print(f"    fold {fold_idx+1}: acc={fold_metrics[-1]['accuracy']:.4f}  f1={fold_metrics[-1]['f1']:.4f}"
              f"  train={tr:.0f}s  eval={ev:.0f}s  trainable={trainable:,}")
        signal("fold_done", config=cfg_index, fold=fold_idx + 1, model=cfg["model"],
               acc=round(fold_metrics[-1]["accuracy"], 4), f1=round(fold_metrics[-1]["f1"], 4),
               train_sec=round(tr))

    if not fold_metrics:
        raise RuntimeError("All folds failed — cannot compute CV metrics")

    mean_train_sec = float(np.mean(fold_train_sec)) if fold_train_sec else 0.0
    mean_eval_sec  = float(np.mean(fold_eval_sec))  if fold_eval_sec  else 0.0

    return (
        {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        mean_train_sec,
        mean_eval_sec,
        fold_models,
    )


def write_row(writer, cfg, mean, std, mean_train_sec, mean_eval_sec, config_idx, total):
    row = {
        "config": config_idx,
        "model": cfg["model"],
        "unfreeze_blocks": cfg["unfreeze_blocks"],
        "lr": cfg["lr"],
        "scheduler": cfg["scheduler"],
        "label_smoothing": cfg["label_smoothing"],
        "weight_decay": cfg["weight_decay"],
        "llrd_factor": cfg.get("llrd_factor", 1.0),
        "color_jitter": cfg.get("use_color_jitter", False),
        "randaugment": cfg.get("use_randaugment", False),
        "random_erasing": cfg.get("use_random_erasing", False),
        "collator": cfg.get("collator", "none"),
        "lora_r": cfg.get("lora_r", 0),
        "acc_mean": f"{mean['accuracy']:.4f}",
        "acc_std":  f"{std['accuracy']:.4f}",
        "f1_mean":  f"{mean['f1']:.4f}",
        "f1_std":   f"{std['f1']:.4f}",
        "prec_mean": f"{mean['precision']:.4f}",
        "rec_mean":  f"{mean['recall']:.4f}",
        "train_sec": f"{mean_train_sec:.1f}",
        "eval_sec":  f"{mean_eval_sec:.1f}",
    }
    writer.writerow(row)


def main():
    set_seed(SEED)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = gpu.total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        print(f"GPU: {gpu.name} ({total_gb:.1f} GB, {GPU_MEMORY_FRACTION*100:.0f}% = {total_gb*GPU_MEMORY_FRACTION:.1f} GB)")
    else:
        print("WARNING: No GPU — running on CPU")

    full_dataset = load_dataset("imagefolder", data_files={"train": os.path.join(TRAIN_DIR, "**")})["train"]
    class_names  = full_dataset.features["label"].names
    num_labels   = len(class_names)
    label2id     = {label: str(i) for i, label in enumerate(class_names)}
    id2label     = {str(i): label for i, label in enumerate(class_names)}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(range(len(full_dataset)), full_dataset["label"]))

    model_lookup = {cfg["name"]: cfg for cfg in MODELS}

    fieldnames = ["config", "model", "unfreeze_blocks", "lr", "scheduler",
                  "label_smoothing", "weight_decay", "llrd_factor",
                  "color_jitter", "randaugment", "random_erasing", "collator", "lora_r",
                  "acc_mean", "acc_std", "f1_mean", "f1_std", "prec_mean", "rec_mean",
                  "train_sec", "eval_sec"]

    # resume support — skip configs already written
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="") as f:
            for row in csv.DictReader(f):
                completed.add(int(row["config"]))
        print(f"Resuming — {len(completed)} configs already done")

    pending = CONFIGS_TO_RUN - completed if CONFIGS_TO_RUN is not None else None
    if pending is not None:
        print(f"CONFIGS_TO_RUN: {sorted(CONFIGS_TO_RUN)}  |  pending: {sorted(pending)}")

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()

        total = len(SEARCH_CONFIGS)
        for i, cfg in enumerate(SEARCH_CONFIGS):
            if i in completed:
                print(f"[{i+1}/{total}] skipping {cfg['model']} (already done)")
                continue
            if CONFIGS_TO_RUN is not None and i not in CONFIGS_TO_RUN:
                continue  # silently skip configs outside the current run set

            print(f"\n[{i+1}/{total}] {cfg['model']}  unfreeze={cfg['unfreeze_blocks']}  lr={cfg['lr']}  "
                  f"sched={cfg['scheduler']}  ls={cfg['label_smoothing']}  wd={cfg['weight_decay']}")
            signal("config_start", config=i, model=cfg["model"], lr=cfg["lr"],
                   unfreeze=cfg["unfreeze_blocks"], total=total)

            if cfg["model"] not in model_lookup:
                print(f"  [skip] unknown model '{cfg['model']}'")
                continue

            model_cfg = model_lookup[cfg["model"]]
            try:
                mean, std, mean_train_sec, mean_eval_sec, fold_models = run_cv(
                    cfg, model_cfg, full_dataset, fold_splits,
                    class_names, num_labels, label2id, id2label,
                    cfg_index=i,
                )
            except Exception as e:
                signal("error", config=i, model=cfg["model"], msg=str(e)[:60])
                print(f"  [fail] config {i}: {e} — skipping")
                torch.cuda.empty_cache()
                continue

            write_row(writer, cfg, mean, std, mean_train_sec, mean_eval_sec, i, total)
            f.flush()

            # ── generate Kaggle submission CSV ────────────────────────────
            data_root = os.path.dirname(TRAIN_DIR)
            test_dir = os.path.join(data_root, "test")
            if os.path.isdir(test_dir) and fold_models:
                os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
                _ip = model_cfg["get_processor"]()
                _, val_tf = build_transforms(_ip)
                test_ids = sorted(
                    [fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")],
                    key=lambda x: int(x.split(".")[0]),
                )
                image_paths = [os.path.join(test_dir, fn) for fn in test_ids]
                test_preds = predict_test_images(fold_models, image_paths, val_tf)
                sub_path = os.path.join(SUBMISSIONS_DIR, f"config_{i}_{cfg['model']}.csv")
                with open(sub_path, "w", newline="") as sf:
                    writer_sub = csv.writer(sf)
                    writer_sub.writerow(["ID", "Label"])
                    for img_id, pred in zip(test_ids, test_preds):
                        writer_sub.writerow([img_id, int(class_names[int(pred)])])
                print(f"  Submission saved: {sub_path}")
                del fold_models
                torch.cuda.empty_cache()

            print(f"  -> acc={mean['accuracy']:.4f}+/-{std['accuracy']:.4f}  f1={mean['f1']:.4f}+/-{std['f1']:.4f}"
                  f"  train={mean_train_sec:.0f}s/fold  eval={mean_eval_sec:.0f}s/fold")
            signal("config_done", config=i, model=cfg["model"],
                   acc=round(mean["accuracy"], 4), f1=round(mean["f1"], 4))

    signal("search_done", total=total, results=RESULTS_FILE)
    print(f"\nDone. Results saved to {RESULTS_FILE}")

    # ── Summary of configs run this session ───────────────────────────────────
    if CONFIGS_TO_RUN is not None and os.path.exists(RESULTS_FILE):
        ran = []
        with open(RESULTS_FILE, newline="") as f_in:
            for row in csv.DictReader(f_in):
                if int(row["config"]) in CONFIGS_TO_RUN:
                    ran.append(row)
        if ran:
            ran.sort(key=lambda r: float(r["acc_mean"]), reverse=True)
            print(f"\n{'='*74}")
            print(f"  Results for this run ({len(ran)} configs), sorted by acc")
            print(f"{'='*74}")
            print(f"  {'cfg':>3}  {'model':<14} {'unfrz':>5} {'lr':>7} {'llrd':>5} {'extras':<16} {'acc':>7} {'f1':>7}")
            print(f"  {'-'*70}")
            for r in ran:
                extras = []
                if r.get("randaugment") == "True": extras.append("rand")
                if r.get("color_jitter") == "True": extras.append("cj")
                if r.get("random_erasing") == "True": extras.append("re")
                if r.get("collator", "none") != "none": extras.append(r["collator"])
                if r.get("lora_r", "0") != "0": extras.append(f"lora_r={r['lora_r']}")
                cfg_entry = SEARCH_CONFIGS[int(r["config"])]
                if cfg_entry.get("num_epochs", NUM_EPOCHS) != NUM_EPOCHS:
                    extras.append(f"ep={cfg_entry.get('num_epochs')}")
                print(f"  {r['config']:>3}  {r['model']:<14} {r['unfreeze_blocks']:>5} {float(r['lr']):>7.0e}"
                      f" {float(r['llrd_factor']):>5.2f}  {' '.join(extras) if extras else '-':<16}"
                      f" {float(r['acc_mean']):>7.4f} {float(r['f1_mean']):>7.4f}")
            print(f"{'='*74}")


if __name__ == "__main__":
    main()
