"""Display and export CV results for pipeline.py."""


_W = 72  # column width for section headers


def build_run_config_lines(cfg: dict) -> list[str]:
    """Build the RUN CONFIGURATION header block from a plain param dict.

    Expected keys (all optional with sensible fallbacks):
        selected_models, num_epochs, batch_size, learning_rate, n_folds, seed,
        unfreeze_blocks, use_lora, lora_r,
        use_randaugment, use_color_jitter, use_random_erasing,
        use_mixup, use_cutmix,
        use_llrd, llrd_factor, use_progressive_unfreeze, progressive_unfreeze_epoch,
        use_tta, tta_augments, use_ensemble
    """
    lora = cfg.get("use_lora", False)
    llrd = cfg.get("use_llrd", False)
    prog = cfg.get("use_progressive_unfreeze", False)
    tta  = cfg.get("use_tta", False)

    training_line = (
        f"  Training        : LLRD={llrd}" +
        (f" (factor={cfg.get('llrd_factor', 0.75)})" if llrd else "") +
        f"  ProgUnfreeze={prog}" +
        (f" (ep{cfg.get('progressive_unfreeze_epoch', 3)})" if prog else "") +
        f"  TTA={tta}" +
        (f" (n={cfg.get('tta_augments', 5)})" if tta else "") +
        f"  Ensemble={cfg.get('use_ensemble', False)}"
    )

    return [
        "=" * _W,
        "RUN CONFIGURATION",
        "=" * _W,
        f"  Models          : {', '.join(cfg.get('selected_models', []))}",
        f"  Epochs          : {cfg.get('num_epochs', '?')}  |  "
        f"Batch: {cfg.get('batch_size', '?')}  |  "
        f"Base LR: {cfg.get('learning_rate', '?')}  |  "
        f"Folds: {cfg.get('n_folds', '?')}  |  "
        f"Seed: {cfg.get('seed', '?')}",
        f"  Unfreeze blocks : {cfg.get('unfreeze_blocks', '?')}  |  LoRA: {lora}" +
        (f" (r={cfg.get('lora_r', 8)})" if lora else ""),
        f"  Augmentation    : RandAugment={cfg.get('use_randaugment', False)}"
        f"  ColorJitter={cfg.get('use_color_jitter', False)}"
        f"  RandomErasing={cfg.get('use_random_erasing', False)}",
        f"  Mix / labels    : Mixup={cfg.get('use_mixup', False)}"
        f"  CutMix={cfg.get('use_cutmix', False)}"
        f"  LabelSmoothing=0.05  WeightDecay=0.01",
        training_line,
        "=" * _W,
    ]


def print_results(results: dict, class_names: list, num_labels: int,
                  n_folds: int, run_cfg_lines: list[str]) -> None:
    """Print run config block, model comparison table, and per-class F1 table."""
    model_names = list(results.keys())

    # ── run config ────────────────────────────────────────────────────
    print()
    for line in run_cfg_lines:
        print(line)

    # ── comparison table ───────────────────────────────────────────────
    print(f"\n{'='*_W}")
    print(f"MODEL COMPARISON ({n_folds}-fold CV, mean ± std)")
    print(f"{'='*_W}")
    print(f"{'model':>14} {'accuracy':>16} {'precision':>16} {'recall':>16} {'f1':>16}")
    print("-" * 80)
    for name in model_names:
        m, s = results[name]["mean"], results[name]["std"]
        print(
            f"{name:>14} {m['accuracy']:.4f}±{s['accuracy']:.4f}  "
            f"{m['precision']:.4f}±{s['precision']:.4f}  "
            f"{m['recall']:.4f}±{s['recall']:.4f}  "
            f"{m['f1']:.4f}±{s['f1']:.4f}"
        )

    best = max(model_names, key=lambda n: results[n]["mean"]["accuracy"])
    print(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)")

    # ── per-class F1 table ─────────────────────────────────────────────
    col_w = 14
    print(f"\n{'='*_W}")
    print(f"PER-CLASS F1 (mean across {n_folds} folds)")
    print(f"{'='*_W}")
    for name in model_names:
        got = sum(1 for cls in class_names if results[name]["per_class"][cls]["f1"] > 0)
        print(f"  {name}: {got}/{num_labels} classes with F1 > 0")
    print()
    print(f"{'class':<25}" + "".join(f"{n:>{col_w}}" for n in model_names))
    print("-" * (25 + col_w * len(model_names)))
    for cls in class_names:
        row = f"{cls:<25}"
        for name in model_names:
            row += f"{results[name]['per_class'][cls]['f1']:>{col_w}.4f}"
        print(row)


def export_results(path: str, results: dict, class_names: list, num_labels: int,
                   n_folds: int, run_cfg_lines: list[str]) -> None:
    """Write run config + comparison table + per-class F1 to a text file."""
    model_names = list(results.keys())
    col_w = 14

    lines = [line + "\n" for line in run_cfg_lines]
    lines += [
        "\n",
        f"MODEL COMPARISON ({n_folds}-fold CV, mean ± std)\n",
        f"{'model':>14} {'accuracy':>16} {'precision':>16} {'recall':>16} {'f1':>16}\n",
        "-" * 80 + "\n",
    ]
    for name in model_names:
        m, s = results[name]["mean"], results[name]["std"]
        lines.append(
            f"{name:>14} {m['accuracy']:.4f}±{s['accuracy']:.4f}  "
            f"{m['precision']:.4f}±{s['precision']:.4f}  "
            f"{m['recall']:.4f}±{s['recall']:.4f}  "
            f"{m['f1']:.4f}±{s['f1']:.4f}\n"
        )

    best = max(model_names, key=lambda n: results[n]["mean"]["accuracy"])
    lines.append(f"\nBest model: {best} (mean val acc {results[best]['mean']['accuracy'] * 100:.2f}%)\n")

    lines.append(f"\n\nPER-CLASS F1 (mean across {n_folds} folds)\n")
    for name in model_names:
        got = sum(1 for cls in class_names if results[name]["per_class"][cls]["f1"] > 0)
        lines.append(f"  {name}: {got}/{num_labels} classes with F1 > 0\n")
    lines.append("\n")
    lines.append(f"{'class':<25}" + "".join(f"{n:>{col_w}}" for n in model_names) + "\n")
    lines.append("-" * (25 + col_w * len(model_names)) + "\n")
    for cls in class_names:
        row = f"{cls:<25}"
        for name in model_names:
            row += f"{results[name]['per_class'][cls]['f1']:>{col_w}.4f}"
        lines.append(row + "\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nResults saved to {path}")
