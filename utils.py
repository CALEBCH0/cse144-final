import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback


# ── Freeze / unfreeze ─────────────────────────────────────────────────────────

def apply_freeze(model, model_name, unfreeze_blocks):
    """Freeze all params then selectively unfreeze top N blocks/stages + classifier.

    Transformers (dinov2, vit): unfreeze_blocks out of 12 encoder layers from top.
    ConvNeXt: unfreeze_blocks out of 4 stages from top.
    unfreeze_blocks=0 → frozen backbone, head only.
    """
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_blocks > 0:
        if model_name == "convnext":
            for name, param in model.named_parameters():
                for i in range(4 - unfreeze_blocks, 4):
                    if f"encoder.stages.{i}." in name:
                        param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                for i in range(12 - unfreeze_blocks, 12):
                    if f"encoder.layer.{i}." in name or f"layers.{i}." in name:
                        param.requires_grad = True

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Layer-wise LR decay (LLRD) ────────────────────────────────────────────────

def get_llrd_optimizer(model, model_name, base_lr, decay_factor, weight_decay):
    """AdamW with per-layer learning rates that decay toward earlier layers.

    Classifier head gets base_lr. Each layer below it is multiplied by
    decay_factor, so shallow layers train more slowly and preserve pretrained
    features. Typical decay_factor: 0.65–0.85.
    """
    groups = []

    if model_name in ("dinov2", "vit"):
        # embeddings — lowest LR
        emb_params = [p for n, p in model.named_parameters()
                      if "embeddings" in n and p.requires_grad]
        if emb_params:
            groups.append({"params": emb_params, "lr": base_lr * (decay_factor ** 12)})

        # 12 encoder layers: layer 0 (shallowest) → layer 11 (deepest)
        for i in range(12):
            layer_params = [p for n, p in model.named_parameters()
                            if (f"encoder.layer.{i}." in n or f"layers.{i}." in n)
                            and p.requires_grad]
            if layer_params:
                groups.append({"params": layer_params,
                                "lr": base_lr * (decay_factor ** (11 - i))})

    elif model_name == "convnext":
        emb_params = [p for n, p in model.named_parameters()
                      if "embeddings" in n and p.requires_grad]
        if emb_params:
            groups.append({"params": emb_params, "lr": base_lr * (decay_factor ** 4)})

        for i in range(4):
            stage_params = [p for n, p in model.named_parameters()
                            if f"encoder.stages.{i}." in n and p.requires_grad]
            if stage_params:
                groups.append({"params": stage_params,
                                "lr": base_lr * (decay_factor ** (3 - i))})

    # classifier head always gets the full base_lr
    head_params = [p for n, p in model.named_parameters()
                   if "classifier" in n and p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr})

    for g in groups:
        g["weight_decay"] = weight_decay

    return torch.optim.AdamW(groups)


# ── Mixup ─────────────────────────────────────────────────────────────────────

class MixupCollator:
    """Replaces the standard collator. Blends pairs of images and creates
    soft one-hot labels for cross-entropy.  alpha=0 disables mixup."""

    def __init__(self, num_classes, alpha=0.2):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])

        if self.alpha <= 0 or not torch.is_grad_enabled():
            return {"pixel_values": pixel_values, "labels": labels}

        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(pixel_values.size(0))

        mixed = lam * pixel_values + (1 - lam) * pixel_values[idx]

        y_a = F.one_hot(labels, self.num_classes).float()
        y_b = F.one_hot(labels[idx], self.num_classes).float()
        soft_labels = lam * y_a + (1 - lam) * y_b

        return {"pixel_values": mixed, "labels": soft_labels}


class CustomTrainer(Trainer):
    """Trainer that supports LLRD and soft labels (Mixup).

    Extra constructor args:
        llrd_factor  (float): decay per layer; 1.0 = disabled (uniform LR).
        model_name   (str):   model key used by get_llrd_optimizer.
        use_mixup    (bool):  if True, expects float soft labels from MixupCollator.
    """

    def __init__(self, *args, llrd_factor=1.0, model_name="", use_mixup=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_factor = llrd_factor
        self.model_name_for_llrd = model_name
        self.use_mixup = use_mixup

    def create_optimizer(self):
        if self.llrd_factor < 1.0:
            self.optimizer = get_llrd_optimizer(
                self.model,
                self.model_name_for_llrd,
                self.args.learning_rate,
                self.llrd_factor,
                self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if labels.dtype == torch.float:
            # soft labels from Mixup — use manual cross-entropy
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Progressive unfreezing ────────────────────────────────────────────────────

class ProgressiveUnfreezeCallback(TrainerCallback):
    """Unfreezes the full backbone after `unfreeze_epoch` epochs have completed.

    Before that epoch the backbone stays frozen (only the classifier trains),
    giving the head time to stabilise before the backbone starts moving.
    """

    def __init__(self, model_name, unfreeze_epoch):
        self.model_name = model_name
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self._unfrozen and state.epoch >= self.unfreeze_epoch:
            for param in model.parameters():
                param.requires_grad = True
            self._unfrozen = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  [ProgressiveUnfreeze] epoch {state.epoch:.0f}: "
                  f"full backbone unfrozen — {trainable:,} trainable params")


# ── Test-time augmentation (TTA) ──────────────────────────────────────────────

def predict_with_tta(trainer, base_dataset, train_tf, n_augments=5):
    """Run inference n_augments times with random train augmentations and
    average the softmax probabilities.  Returns numpy array (N, num_classes)."""

    tta_dataset = base_dataset.with_transform(train_tf)
    all_probs = []

    for _ in range(n_augments):
        out = trainer.predict(tta_dataset)
        probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(models, dataset, collate_fn, batch_size=32):
    """Average logits from a list of models over the same dataset.
    Returns numpy array (N, num_classes) of averaged logits."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    all_logits = []
    for model in models:
        model.eval()
        model.to(device)
        fold_logits = []
        with torch.no_grad():
            for batch in loader:
                pv = batch["pixel_values"].to(device)
                out = model(pixel_values=pv)
                fold_logits.append(out.logits.cpu())
        all_logits.append(torch.cat(fold_logits, dim=0))
        model.cpu()
        torch.cuda.empty_cache()

    return torch.stack(all_logits).mean(0).numpy()


# ── CutMix ────────────────────────────────────────────────────────────────────

class CutMixCollator:
    """Pastes a random rectangular patch from one image into another and
    mixes labels proportional to the actual cut area.  alpha=0 disables."""

    def __init__(self, num_classes, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])

        if self.alpha <= 0 or not torch.is_grad_enabled():
            return {"pixel_values": pixel_values, "labels": labels}

        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(pixel_values.size(0))

        _, _, H, W = pixel_values.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        mixed = pixel_values.clone()
        mixed[:, :, y1:y2, x1:x2] = pixel_values[idx, :, y1:y2, x1:x2]

        # Recompute lam from actual box size so label mix is accurate
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

        y_a = F.one_hot(labels, self.num_classes).float()
        y_b = F.one_hot(labels[idx], self.num_classes).float()
        soft_labels = lam * y_a + (1 - lam) * y_b

        return {"pixel_values": mixed, "labels": soft_labels}


class MixupCutMixCollator:
    """Randomly applies Mixup or CutMix per batch (50/50 by default)."""

    def __init__(self, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, mixup_prob=0.5):
        self.mixup = MixupCollator(num_classes, alpha=mixup_alpha)
        self.cutmix = CutMixCollator(num_classes, alpha=cutmix_alpha)
        self.mixup_prob = mixup_prob

    def __call__(self, examples):
        if np.random.random() < self.mixup_prob:
            return self.mixup(examples)
        return self.cutmix(examples)


# ── LoRA ──────────────────────────────────────────────────────────────────────

# Target modules verified by inspecting named_modules() of each loaded model:
#   DINOv2 (facebook/dinov2-base):    query, value  in dinov2.encoder.layer.{i}.attention.attention.*
#   timm ViT (vit_base_patch16_224):  qkv           in timm_model.blocks.{i}.attn.qkv
_LORA_TARGETS = {
    "dinov2": {"target_modules": ["query", "value"], "modules_to_save": ["classifier"]},
    "vit":    {"target_modules": ["qkv"],            "modules_to_save": ["head"]},
}


def apply_lora(model, model_name, r=8, lora_alpha=16, lora_dropout=0.05):
    """Wrap model with LoRA adapters. Backbone is frozen by PEFT; only LoRA
    weights and the task head remain trainable.  Returns the PEFT model."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("Install peft: pip install 'peft>=0.6.0'")

    if model_name not in _LORA_TARGETS:
        raise ValueError(
            f"LoRA not configured for '{model_name}'. "
            f"Supported: {list(_LORA_TARGETS)}. "
            "Add target_modules to _LORA_TARGETS in utils.py."
        )

    cfg = _LORA_TARGETS[model_name]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=cfg["target_modules"],
        modules_to_save=cfg["modules_to_save"],
        bias="none",
    )
    return get_peft_model(model, lora_config)
