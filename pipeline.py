import argparse
import csv
import os
import subprocess
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, set_seed

from config import (
    BATCH_SIZE,
    GPU_MEMORY_FRACTION,
    LEARNING_RATE,
    LLRD_FACTOR,
    LORA_R,
    N_FOLDS,
    NUM_EPOCHS,
    PROGRESSIVE_UNFREEZE_EPOCH,
    SEED,
    SELECTED_MODELS,
    TRAIN_DIR,
    TTA_AUGMENTS,
    UNFREEZE_BLOCKS,
    USE_COLOR_JITTER,
    USE_CUTMIX,
    USE_ENSEMBLE,
    USE_LLRD,
    USE_LORA,
    USE_MIXUP,
    USE_PROGRESSIVE_UNFREEZE,
    USE_RANDAUGMENT,
    USE_RANDOM_ERASING,
    USE_TTA,
)
from models import MODELS
from resulter import build_run_config_lines, export_results, print_results
from transforms import build_transforms
from utils import (
    CustomTrainer,
    CutMixCollator,
    EpochAccuracyCallback,
    MixupCollator,
    MixupCutMixCollator,
    ProgressiveUnfreezeCallback,
    apply_freeze,
    apply_lora,
    ensemble_predict,
    predict_test_images,
    predict_with_tta,
    signal,
)


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
    # parser.add_argument("--export", default="pipeline_results.txt", type=str, metavar="FILE", help="Save results to a text file")
    parser.add_argument("--export", type=str, metavar="FILE", help="Save results to a text file")
    parser.add_argument("--submit", action="store_true", help="Submit generated submission CSV to Kaggle after training")
    parser.add_argument("--message", type=str, default="pipeline run", metavar="MSG", help="Kaggle submission message (used with --submit)")
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
        signal("model_start", model=name, folds=N_FOLDS)

        image_processor = model_cfg["get_processor"]()
        train_tf, val_tf = build_transforms(
            image_processor,
            use_color_jitter=USE_COLOR_JITTER,
            use_randaugment=USE_RANDAUGMENT,
            use_random_erasing=USE_RANDOM_ERASING,
        )

        if USE_MIXUP and USE_CUTMIX:
            data_collator = MixupCutMixCollator(num_labels, mixup_alpha=0.2, cutmix_alpha=1.0)
        elif USE_CUTMIX:
            data_collator = CutMixCollator(num_labels, alpha=1.0)
        elif USE_MIXUP:
            data_collator = MixupCollator(num_labels, alpha=0.2)
        else:
            data_collator = collate_fn

        fold_metrics = []
        fold_train_acc = []
        fold_per_class_f1 = []
        fold_per_class_precision = []
        fold_per_class_recall = []
        fold_models = []
        fold_train_sec = []
        fold_eval_sec = []
        _fmt = lambda s: f"{s/60:.1f}m" if s >= 60 else f"{s:.0f}s"

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            print(f"\n  -- Fold {fold_idx + 1}/{N_FOLDS} --")

            train_fold = full_dataset.select(train_idx).with_transform(train_tf)
            val_fold = full_dataset.select(val_idx).with_transform(val_tf)

            model = model_cfg["get_model"](num_labels, label2id, id2label)

            if USE_LORA:
                model = apply_lora(model, name, r=LORA_R)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  LoRA r={LORA_R} — {trainable:,} trainable params")
            else:
                trainable = apply_freeze(model, name, UNFREEZE_BLOCKS)
                print(f"  Frozen backbone — {trainable:,} trainable params (unfreeze_blocks={UNFREEZE_BLOCKS})")

            train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)

            callbacks = []
            callbacks.append(EpochAccuracyCallback(train_eval_fold, val_fold, collate_fn))
            if USE_PROGRESSIVE_UNFREEZE and not USE_LORA:
                callbacks.append(ProgressiveUnfreezeCallback(name, PROGRESSIVE_UNFREEZE_EPOCH))

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

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                compute_metrics=compute_metrics,
                processing_class=image_processor,
                data_collator=data_collator,
                callbacks=callbacks,
                llrd_factor=LLRD_FACTOR if (USE_LLRD and not USE_LORA) else 1.0,
                model_name=name,
                use_mixup=USE_MIXUP or USE_CUTMIX,
            )

            t0 = time.perf_counter()
            trainer.train()
            train_sec = time.perf_counter() - t0

            trainer.data_collator = collate_fn  # plain collator — no mixup on val/train eval
            t0 = time.perf_counter()
            preds_out = trainer.predict(val_fold)
            eval_sec = time.perf_counter() - t0

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

            # training accuracy — evaluate on train split with val transforms (no aug)
            train_eval_fold = full_dataset.select(train_idx).with_transform(val_tf)
            train_preds_out = trainer.predict(train_eval_fold)
            train_acc = accuracy_score(train_preds_out.label_ids,
                                       np.argmax(train_preds_out.predictions, axis=1))
            fold_train_acc.append(train_acc)

            fold_train_sec.append(train_sec)
            fold_eval_sec.append(eval_sec)

            print(f"  Fold {fold_idx + 1}: train_acc={train_acc:.4f}  val_acc={fold_metrics[-1]['accuracy']:.4f}"
                  f"  f1={fold_metrics[-1]['f1']:.4f}  train={_fmt(train_sec)}  eval={_fmt(eval_sec)}")
            signal("fold_done", model=name, fold=fold_idx + 1, n_folds=N_FOLDS,
                   train_acc=round(train_acc, 4),
                   val_acc=round(fold_metrics[-1]["accuracy"], 4),
                   f1=round(fold_metrics[-1]["f1"], 4),
                   train_sec=round(train_sec))

            if USE_TTA:
                tta_probs = predict_with_tta(trainer, full_dataset.select(val_idx), train_tf, n_augments=TTA_AUGMENTS)
                tta_preds = np.argmax(tta_probs, axis=1)
                tta_acc = accuracy_score(true, tta_preds)
                tta_f1 = f1_score(true, tta_preds, average="macro", zero_division=0)
                print(f"  Fold {fold_idx + 1} TTA:  acc={tta_acc:.4f}  f1={tta_f1:.4f}  (delta acc {tta_acc - fold_metrics[-1]['accuracy']:+.4f})")

            if USE_ENSEMBLE:
                fold_models.append(model)

        mean = {m: float(np.mean([f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}
        std  = {m: float(np.std( [f[m] for f in fold_metrics])) for m in ("accuracy", "precision", "recall", "f1")}

        mean_per_class_f1        = np.mean(fold_per_class_f1, axis=0)
        mean_per_class_precision = np.mean(fold_per_class_precision, axis=0)
        mean_per_class_recall    = np.mean(fold_per_class_recall, axis=0)

        total_train = sum(fold_train_sec)
        total_eval  = sum(fold_eval_sec)

        results[name] = {
            "folds": fold_metrics,
            "mean": mean,
            "std": std,
            "train_acc_mean": float(np.mean(fold_train_acc)),
            "train_acc_std":  float(np.std(fold_train_acc)),
            "per_class": {
                class_names[i]: {
                    "f1": float(mean_per_class_f1[i]),
                    "precision": float(mean_per_class_precision[i]),
                    "recall": float(mean_per_class_recall[i]),
                }
                for i in range(num_labels)
            },
            "total_train_sec": total_train,
            "total_eval_sec": total_eval,
        }
        signal("model_done", model=name, acc=round(mean["accuracy"], 4),
               f1=round(mean["f1"], 4), total_min=round(total_train / 60, 1))
        print(f"\n{name} CV results ({N_FOLDS} folds):")
        print(f"  {'acc':>6}: {mean['accuracy']:.4f} +/-{std['accuracy']:.4f}")
        print(f"  {'prec':>6}: {mean['precision']:.4f} +/-{std['precision']:.4f}")
        print(f"  {'rec':>6}: {mean['recall']:.4f} +/-{std['recall']:.4f}")
        print(f"  {'f1':>6}: {mean['f1']:.4f} +/-{std['f1']:.4f}")
        print(f"  time : train {total_train/60:.1f}m total ({np.mean(fold_train_sec):.0f}s/fold)"
              f"  |  eval {total_eval:.0f}s total ({np.mean(fold_eval_sec):.0f}s/fold)")

        if USE_ENSEMBLE and len(fold_models) > 1:
            print(f"\n  Running ensemble ({len(fold_models)} fold models) on full dataset...")
            ens_logits = ensemble_predict(fold_models, full_dataset.with_transform(val_tf), collate_fn)
            ens_preds = np.argmax(ens_logits, axis=1)
            ens_labels = np.array(full_dataset["label"])
            ens_acc = accuracy_score(ens_labels, ens_preds)
            ens_f1 = f1_score(ens_labels, ens_preds, average="macro", zero_division=0)
            print(f"  Ensemble (full train set): acc={ens_acc:.4f}  f1={ens_f1:.4f}")

        # ── Kaggle test inference ──────────────────────────────────────────────
        submission_models = fold_models if fold_models else [model]
        data_root = os.path.dirname(TRAIN_DIR)
        test_dir = os.path.join(data_root, "test")

        if os.path.isdir(test_dir):
            # Use all jpg files — Kaggle expects every image in the test dir (1036, not just 1000)
            test_ids = sorted(
                [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")],
                key=lambda x: int(x.split(".")[0]),
            )

            image_paths = [os.path.join(test_dir, img_id) for img_id in test_ids]
            submission_path = f"submission_{name}.csv"
            print(f"\n  Predicting {len(test_ids)} test images ({len(submission_models)} fold model(s))...")
            test_preds = predict_test_images(submission_models, image_paths, val_tf)

            with open(submission_path, "w", newline="") as sf:
                writer = csv.writer(sf)
                writer.writerow(["ID", "Label"])
                for img_id, pred in zip(test_ids, test_preds):
                    # class_names are alphabetically sorted folder names ("0","1","10",...)
                    # convert internal index back to the folder name (= Kaggle's integer label)
                    writer.writerow([img_id, int(class_names[int(pred)])])
            print(f"  Submission saved: {submission_path}")

            if args.submit:
                print(f"  Submitting to Kaggle...")
                result = subprocess.run(
                    [
                        "kaggle", "competitions", "submit",
                        "-c", "ucsc-cse-144-spring-2026-final-project",
                        "-f", submission_path,
                        "-m", args.message,
                    ],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    print(f"  {result.stdout.strip()}")
                else:
                    print(f"  Kaggle submission failed: {result.stderr.strip()}")

    run_cfg = {
        "selected_models": SELECTED_MODELS,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "unfreeze_blocks": UNFREEZE_BLOCKS,
        "use_lora": USE_LORA,
        "lora_r": LORA_R,
        "use_randaugment": USE_RANDAUGMENT,
        "use_color_jitter": USE_COLOR_JITTER,
        "use_random_erasing": USE_RANDOM_ERASING,
        "use_mixup": USE_MIXUP,
        "use_cutmix": USE_CUTMIX,
        "use_llrd": USE_LLRD,
        "llrd_factor": LLRD_FACTOR,
        "use_progressive_unfreeze": USE_PROGRESSIVE_UNFREEZE,
        "progressive_unfreeze_epoch": PROGRESSIVE_UNFREEZE_EPOCH,
        "use_tta": USE_TTA,
        "tta_augments": TTA_AUGMENTS,
        "use_ensemble": USE_ENSEMBLE,
    }
    run_cfg_lines = build_run_config_lines(run_cfg)
    print_results(results, class_names, num_labels, N_FOLDS, run_cfg_lines)

    best_model = max(results, key=lambda n: results[n]["mean"]["accuracy"]) if results else "none"
    best_acc = results[best_model]["mean"]["accuracy"] if results else 0.0
    signal("pipeline_done", best=best_model, acc=round(best_acc, 4), models=len(results))

    if args.export:
        export_results(args.export, results, class_names, num_labels, N_FOLDS, run_cfg_lines)


if __name__ == "__main__":
    main()
