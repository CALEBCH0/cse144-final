import os

import numpy as np
import torch
from datasets import load_dataset
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

SEED = 42
TRAIN_DIR = "data/train"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.15


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

    def train_transforms(batch):
        batch["pixel_values"] = [_train_tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    def val_transforms(batch):
        batch["pixel_values"] = [_val_tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    return train_transforms, val_transforms


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    set_seed(SEED)

    dataset = load_dataset("imagefolder", data_files={"train": os.path.join(TRAIN_DIR, "**")})
    split = dataset["train"].train_test_split(test_size=VAL_SPLIT, seed=SEED)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": float((preds == p.label_ids).mean())}

    results = {}

    for model_cfg in MODELS:
        name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Training: {name} ({model_cfg['model_id']})")
        print(f"{'='*60}")

        image_processor = model_cfg["get_processor"]()
        model = model_cfg["get_model"](num_labels, label2id, id2label)

        train_tf, val_tf = build_transforms(image_processor)
        train_dataset = dataset["train"].with_transform(train_tf)
        val_dataset = dataset["validation"].with_transform(val_tf)

        training_args = TrainingArguments(
            output_dir=model_cfg["output_dir"],
            do_train=True,
            do_eval=True,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            seed=SEED,
            remove_unused_columns=False,
            logging_steps=10,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            processing_class=image_processor,
            data_collator=collate_fn,
        )

        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model()

        val_accuracy = metrics["eval_accuracy"]
        results[name] = val_accuracy
        print(f"{name} best validation accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

    print(f"\n{'='*60}")
    print("MODEL COMPARISON - FINAL VALIDATION ACCURACIES")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:>10}: {acc:.4f} ({acc * 100:.2f}%)")
    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best] * 100:.2f}%)")


if __name__ == "__main__":
    main()
