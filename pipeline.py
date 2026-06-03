import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import TimmWrapperImageProcessor, Trainer, TrainingArguments, set_seed

from models import MODELS

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


class _Transform:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, batch):
        batch["pixel_values"] = [self.tf(img.convert("RGB")) for img in batch["image"]]
        return batch


def build_transforms(image_processor):
    if isinstance(image_processor, TimmWrapperImageProcessor):
        _train_tf = image_processor.train_transforms
        _val_tf = image_processor.val_transforms
    else:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        else:
            normalize = Lambda(lambda x: x)

        _train_tf = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
        _val_tf = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    return _Transform(_train_tf), _Transform(_val_tf)


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    true = p.label_ids
    return {
        "accuracy": accuracy_score(true, preds),
        "precision": precision_score(true, preds, average="macro", zero_division=0),
        "recall": recall_score(true, preds, average="macro", zero_division=0),
        "f1": f1_score(true, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, metavar="FILE", help="Save results to a text file")
    args = parser.parse_args()

    set_seed(SEED)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_gb = gpu.total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        print(f"GPU: {gpu.name} ({total_gb:.1f} GB, using {GPU_MEMORY_FRACTION*100:.0f}% = {total_gb * GPU_MEMORY_FRACTION:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU detected — training will run on CPU")

    full_dataset = load_dataset("imagefolder", data_files={"train": os.path.join(TRAIN_DIR, "**")})["train"]

    class_names = full_dataset.features["label"].names
    num_labels = len(class_names)
    label2id = {label: str(i) for i, label in enumerate(class_names)}
    id2label = {str(i): label for i, label in enumerate(class_names)}
    all_label_ids = list(range(num_labels))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(range(len(full_dataset)), full_dataset["label"]))

    results = {}
    selected = [cfg for cfg in MODELS if cfg["name"] in SELECTED_MODELS]

    for model_cfg in selected:
        name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Training: {name} ({model_cfg['model_id']})  [{N_FOLDS}-fold CV]")
        print(f"{'='*60}")

        image_processor = model_cfg["get_processor"]()
        train_tf, val_tf = build_transforms(image_processor)

        fold_metrics = []
        fold_per_class_f1 = []
        fold_per_class_precision = []
        fold_per_class_recall = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            print(f"\n  -- Fold {fold_idx + 1}/{N_FOLDS} --")

            train_fold = full_dataset.select(train_idx).with_transform(train_tf)
            val_fold = full_dataset.select(val_idx).with_transform(val_tf)

            model = model_cfg["get_model"](num_labels, label2id, id2label)

            if model_cfg.get("freeze_backbone"):
                for pname, param in model.named_parameters():
                    if "classifier" not in pname:
                        param.requires_grad = False
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Frozen backbone — {trainable:,} trainable params")

            training_args = TrainingArguments(
                output_dir=os.path.join(model_cfg["output_dir"], f"fold_{fold_idx}"),
                do_train=True,
                do_eval=True,
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                learning_rate=model_cfg.get("learning_rate", LEARNING_RATE),
                weight_decay=0.01,
                label_smoothing_factor=0.05,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                seed=SEED,
                remove_unused_columns=False,
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,
                dataloader_pin_memory=torch.cuda.is_available(),
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                compute_metrics=compute_metrics,
                processing_class=image_processor,
                data_collator=collate_fn,
            )

            trainer.train()

            preds_out = trainer.predict(val_fold)
            preds = np.argmax(preds_out.predictions, axis=1)
            true = preds_out.label_ids

            fold_metrics.append({
                "accuracy": accuracy_score(true, preds),
                "precision": precision_score(true, preds, average="macro", zero_division=0),
                "recall": recall_score(true, preds, average="macro", zero_division=0),
                "f1": f1_score(true, preds, average="macro", zero_division=0),
            })
            fold_per_class_f1.append(f1_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
            fold_per_class_precision.append(precision_score(true, preds, labels=all_label_ids, average=None, zero_division=0))
            fold_per_class_recall.append(recall_score(true, preds, labels=all_label_ids, average=None, zero_division=0))

            print(f"  Fold {fold_idx + 1} val: acc={fold_metrics[-1]['accuracy']:.4f}  f1={fold_metrics[-1]['f1']:.4f}")

        mean = {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}
        std  = {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}

        mean_per_class_f1        = np.mean(fold_per_class_f1, axis=0)
        mean_per_class_precision = np.mean(fold_per_class_precision, axis=0)
        mean_per_class_recall    = np.mean(fold_per_class_recall, axis=0)

        results[name] = {
            "folds": fold_metrics,
            "mean": mean,
            "std": std,
            "per_class": {
                class_names[i]: {
                    "f1": float(mean_per_class_f1[i]),
                    "precision": float(mean_per_class_precision[i]),
                    "recall": float(mean_per_class_recall[i]),
                }
                for i in range(num_labels)
            },
        }

        print(f"\n{name} CV results ({N_FOLDS} folds):")
        print(f"  {'acc':>6}: {mean['accuracy']:.4f} ± {std['accuracy']:.4f}")
        print(f"  {'prec':>6}: {mean['precision']:.4f} ± {std['precision']:.4f}")
        print(f"  {'rec':>6}: {mean['recall']:.4f} ± {std['recall']:.4f}")
        print(f"  {'f1':>6}: {mean['f1']:.4f} ± {std['f1']:.4f}")

    # ── comparison table ──────────────────────────────────────────────
    model_names = list(results.keys())
    print(f"\n{'='*72}")
    print(f"MODEL COMPARISON ({N_FOLDS}-fold CV, mean ± std)")
    print(f"{'='*72}")
    print(f"{'model':>14} {'accuracy':>16} {'precision':>16} {'recall':>16} {'f1':>16}")
    print("-" * 80)
    for n in model_names:
        m, s = results[n]["mean"], results[n]["std"]
        print(f"{n:>14} {m['accuracy']:.4f}±{s['accuracy']:.4f}  {m['precision']:.4f}±{s['precision']:.4f}  {m['recall']:.4f}±{s['recall']:.4f}  {m['f1']:.4f}±{s['f1']:.4f}")

    best = max(model_names, key=lambda n: results[n]["mean"]["accuracy"])
    print(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)")

    # ── per-class table ───────────────────────────────────────────────
    col_w = 14
    print(f"\n{'='*72}")
    print(f"PER-CLASS F1 (mean across {N_FOLDS} folds)")
    print(f"{'='*72}")
    for n in model_names:
        got_right = sum(1 for cls in class_names if results[n]["per_class"][cls]["f1"] > 0)
        print(f"  {n}: {got_right}/{num_labels} classes with F1 > 0")
    print()
    print(f"{'class':<25}" + "".join(f"{n:>{col_w}}" for n in model_names))
    print("-" * (25 + col_w * len(model_names)))
    for cls in class_names:
        row = f"{cls:<25}"
        for n in model_names:
            row += f"{results[n]['per_class'][cls]['f1']:>{col_w}.4f}"
        print(row)

    # ── export ────────────────────────────────────────────────────────
    if args.export:
        lines = [
            f"MODEL COMPARISON ({N_FOLDS}-fold CV, mean ± std)\n",
            f"{'model':>14} {'accuracy':>16} {'precision':>16} {'recall':>16} {'f1':>16}\n",
            "-" * 80 + "\n",
        ]
        for n in model_names:
            m, s = results[n]["mean"], results[n]["std"]
            lines.append(f"{n:>14} {m['accuracy']:.4f}±{s['accuracy']:.4f}  {m['precision']:.4f}±{s['precision']:.4f}  {m['recall']:.4f}±{s['recall']:.4f}  {m['f1']:.4f}±{s['f1']:.4f}\n")
        lines.append(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)\n")
        lines.append(f"\n\nPER-CLASS F1 (mean across {N_FOLDS} folds)\n")
        for n in model_names:
            got_right = sum(1 for cls in class_names if results[n]["per_class"][cls]["f1"] > 0)
            lines.append(f"  {n}: {got_right}/{num_labels} classes with F1 > 0\n")
        lines.append("\n")
        lines.append(f"{'class':<25}" + "".join(f"{n:>{col_w}}" for n in model_names) + "\n")
        lines.append("-" * (25 + col_w * len(model_names)) + "\n")
        for cls in class_names:
            row = f"{cls:<25}"
            for n in model_names:
                row += f"{results[n]['per_class'][cls]['f1']:>{col_w}.4f}"
            lines.append(row + "\n")
        with open(args.export, "w") as f:
            f.writelines(lines)
        print(f"\nResults saved to {args.export}")


if __name__ == "__main__":
    main()
