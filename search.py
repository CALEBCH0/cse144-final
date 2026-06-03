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
    MixupCollator,
    MixupCutMixCollator,
    apply_freeze as _apply_freeze,
    apply_lora,
)

RESULTS_FILE = "search_results.csv"

# ── Configs to run ────────────────────────────────────────────────────────────
# Set to None to run everything not yet in the CSV (full resume mode).
# Set to a set of indices to run only those specific configs.
# Completed configs (already in search_results.csv) are always skipped regardless.
CONFIGS_TO_RUN = {
    # ViT LR sweep (linear)
    12, 13, 14, 15,
    # ViT scheduler (cosine)
    18, 19,
    # ViT weight decay
    24,
    # ViT LLRD factors
    28, 29, 30,
    # Aug + collator combos (full aug stack + mixup / cutmix / mixup_cutmix)
    35, 36, 37,
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
]
# fmt: on

NUM_EPOCHS = 10
BATCH_SIZE = 32


def run_cv(cfg, model_cfg, full_dataset, fold_splits, class_names, num_labels, label2id, id2label):
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

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_fold = full_dataset.select(train_idx).with_transform(train_tf)
        val_fold   = full_dataset.select(val_idx).with_transform(val_tf)

        model = model_cfg["get_model"](num_labels, label2id, id2label)

        lora_r = cfg.get("lora_r", 0)
        if lora_r > 0:
            model = apply_lora(model, cfg["model"], r=lora_r)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            trainable = _apply_freeze(model, cfg["model"], cfg["unfreeze_blocks"])

        training_args = TrainingArguments(
            output_dir=os.path.join(model_cfg["output_dir"], "search", f"fold_{fold_idx}"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
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
        torch.cuda.empty_cache()
        tr = fold_train_sec[-1] if fold_train_sec else 0
        ev = fold_eval_sec[-1] if fold_eval_sec else 0
        print(f"    fold {fold_idx+1}: acc={fold_metrics[-1]['accuracy']:.4f}  f1={fold_metrics[-1]['f1']:.4f}"
              f"  train={tr:.0f}s  eval={ev:.0f}s  trainable={trainable:,}")

    if not fold_metrics:
        raise RuntimeError("All folds failed — cannot compute CV metrics")

    mean_train_sec = float(np.mean(fold_train_sec)) if fold_train_sec else 0.0
    mean_eval_sec  = float(np.mean(fold_eval_sec))  if fold_eval_sec  else 0.0

    return (
        {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")},
        mean_train_sec,
        mean_eval_sec,
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

            if cfg["model"] not in model_lookup:
                print(f"  ✗ unknown model '{cfg['model']}' — skipping")
                continue

            model_cfg = model_lookup[cfg["model"]]
            try:
                mean, std, mean_train_sec, mean_eval_sec = run_cv(
                    cfg, model_cfg, full_dataset, fold_splits,
                    class_names, num_labels, label2id, id2label,
                )
            except Exception as e:
                print(f"  ✗ config {i} failed: {e} — skipping")
                torch.cuda.empty_cache()
                continue

            write_row(writer, cfg, mean, std, mean_train_sec, mean_eval_sec, i, total)
            f.flush()

            print(f"  → acc={mean['accuracy']:.4f}±{std['accuracy']:.4f}  f1={mean['f1']:.4f}±{std['f1']:.4f}"
                  f"  train={mean_train_sec:.0f}s/fold  eval={mean_eval_sec:.0f}s/fold")

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
