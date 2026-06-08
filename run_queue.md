# Run Queue

Claude reads and updates this file after every run to track state and schedule next steps.
Format: move entry from STAGED → RUNNING (add PID/monitors) → FINISHED (add results/learnings).

---

## RUNNING

_(none)_

---

## STAGED

### run_006 — search.py (CLIP-ViT-Large/14 configs 91–93)
- **Priority**: 3
- **Script**: search.py
- **Change needed**: set `CONFIGS_TO_RUN = {91, 92, 93}` in search.py
- **Log**: search_run.log
- **Models**: clip_vit (openai/clip-vit-large-patch14)
- **Configs**:
  - 91: baseline (10ep, no class_weights)
  - 92: +class_weights (expected to hurt)
  - 93: 20ep, no class_weights
- **ETA**: ~2.5h
- **Notes**: Bug fixed (added `self.config = self.vision_model.config`). Result needed for report section 5.5.22.
- **Depends on**: none

---

## FINISHED

### run_008 — search.py (SigLIP2-SO400M-512 configs 110–112)
- **Started**: 2026-06-08
- **Completed**: 2026-06-08
- **Total time**: ~3h
- **Status**: SUCCESS
- **Results**:
  | Config | Unfreeze | Epochs | Acc | F1 |
  |---|---|---|---|---|
  | 110 | 6 | 20 | 94.07% | 0.9295 |
  | 111 | 6 | 30 | 93.70% | 0.9237 |
  | **112** | **2** | **20** | **94.72%** | **0.9352** |
- **Best config**: 112 (unfreeze=2, cosine, 20ep) → **94.72%**
- **Key findings**:
  - unfreeze=2 is optimal for 512px (same as 384px) — deeper unfreeze hurts
  - Longer training at unfreeze=6 (30ep, cfg111) regresses −0.37% vs 20ep — more overfitting at higher resolution
  - Best 512px (94.72%) < best 384px (95.27%) by **0.55%** — native resolution confirmed optimal
  - DINOv2-518 regression (−9.64%) and SigLIP2-512 regression (−0.55%) share the same root cause: positional embedding interpolation noise + insufficient data for larger token count
  - 512px ensemble with 384px is viable (competitor did ~96.4%+95.4%→97%); test with soft weight search
- **Next action**: soft ensemble weight grid search (SigLIP2-384 + SigLIP2-512); consider SigLIP2-384 + DINOv3 ensemble

### run_007 — search.py (DINOv3 ViT-L/16 configs 113–119)
- **Started**: 2026-06-08
- **Completed**: 2026-06-08
- **Total time**: ~3.5h
- **Status**: SUCCESS
- **Results**:
  | Config | Unfreeze | LR | Epochs | Acc | F1 |
  |---|---|---|---|---|---|
  | 113 | 2 | 1e-4 | 20 | 92.21% | 0.9134 |
  | 114 | 2 | 5e-5 | 20 | 88.88% | 0.8734 |
  | 115 | 4 | 1e-4 | 20 | 92.59% | 0.9149 |
  | 116 | 6 | 1e-4 | 20 | 92.86% | 0.9201 |
  | 117 | 2 | 1e-4 | 50 | 93.60% | 0.9277 |
  | **118** | **4** | **1e-4** | **50** | **93.79%** | **0.9303** |
  | 119 | 4 | 5e-5 | 50 | 93.33% | 0.9259 |
- **Best config**: 118 (unfreeze=4, lr=1e-4, 50ep) → **93.79%**
- **Key findings**:
  - DINOv3 monotonically improves with deeper unfreeze at 20ep (unique — SigLIP2 peaks at unfreeze=2)
  - 50ep adds +1.39–1.58% over 20ep (same pattern as DINOv2-Large)
  - lr=5e-5 recovers at 50ep (93.33%) but still loses to 1e-4 (93.79%)
  - DINOv3 best (93.79%) < SigLIP2 best (95.27%) — 7.5 pt gap; DINOv3 is strong but not competitive with SigLIP2 on this task
  - Next: pipeline run with DINOv3 cfg-118 settings worth considering for ensemble diversity
- **Next action**: run_008 launched (SigLIP2-512 search)

### run_005 — pipeline.py (SigLIP2-SO400M-512 solo validation)
- **Started**: 2026-06-07 ~20:10
- **Completed**: 2026-06-07 ~21:21
- **Total time**: ~71m
- **Status**: SUCCESS
- **Models**: siglip2_so400m_512 (512px, batch_size=8, unfreeze=6, 15ep, lr=1e-4, llrd=0.8)
- **R1 Results**: acc=94.53%, f1=0.9361
- **Key finding**: 512px REGRESSION vs 384px (−0.65%). Do NOT add to ensemble. Higher resolution hurts fine-tuned SigLIP2 — same pattern as DINOv2-518 (−9.6%). Native resolution is optimal.
- **Submission**: submission_siglip2_so400m_512.csv (do not submit — worse than 384px)

### run_005a — pipeline.py (dinov2_large + siglip2, hard routing, min=20, pseudo)
- **Started**: 2026-06-07 ~14:30
- **Completed**: 2026-06-07 ~19:50 (crashed)
- **Total time**: ~5h 20m
- **Status**: CRASHED — siglip2 R2 fold 4 silent OOM (same location as run_004)
- **Crash location**: RAM accumulation over R2 folds; upsampled dataset (1997 images) larger than R1 (1079) → virtual address exhaustion at fold 4 despite `del` fix
- **Config**: SELECTED_MODELS=[dinov2_large, siglip2_so400m], UPSAMPLE_MIN_COUNT=20, USE_PSEUDO_LABEL=True, USE_CLASS_ROUTING=True (hard routing)
- **R1 Results**: dinov2_large=89.71% (new best, +1.30% vs prev 88.41%), siglip2=95.18%
- **Hard routing R1**: 95.09% (siglip2 owns 91/100 classes, dinov2_large owns 9/100)
- **R2 Results**: dinov2_large=87.86% (−1.85% vs R1 — pseudo hurt), siglip2 R2 folds 1–3 only (never completed)
- **Key findings**:
  - Hard routing: 95.09% CV — only 0.09% below SigLIP2 solo; dinov2_large genuinely leads 9 hard classes
  - dinov2_large +1.30% improvement is likely run variance (same config, same seed)
  - Pseudo-labels hurt dinov2_large (−1.85%) — opposite of run_004 (+0.56%); stochastic
  - SigLIP2 pseudo-labels = zero effect again (folds 1–3 identical to R1), consistent across all runs
  - Crash: R2 upsampled dataset is ~1997 images; combined with persistent pseudo-label data still in RAM → OOM at fold 4. Need stronger per-fold cleanup in R2.
- **Submissions saved**: submission_siglip2_so400m.csv (R1 95.18%), submission_routed.csv (hard routing 95.09%), submission_pseudo_dinov2_large.csv (R2 87.86% — do not submit)

### run_004c — pipeline.py (3-model + min=20 + crash fix — confirmed working)
- **Started**: 2026-06-07 ~00:00
- **Completed**: 2026-06-07 ~12:11
- **Total time**: ~12h 11m
- **Status**: SUCCESS — SigLIP2 R2 fold 4 passed (crash fix confirmed for 3-model run)
- **Models**: dinov2, dinov2_large, siglip2_so400m
- **R1 Results**: dinov2≈85.91%, dinov2_large≈88.41%, siglip2≈95.18%
- **R2 Results**: siglip2≈95.27% (+0.09%) ← best R2 ever
- **Kaggle**: submitted submission_pseudo_siglip2_so400m.csv → **95.454%** (5th place)
- **Key finding**: SigLIP2 R2 +0.09% is marginal; consistent with near-zero pseudo-label gain pattern

### run_004b — pipeline.py (SigLIP2-only pseudo + upsampling min=30 + crash fix)
- **Started**: 2026-06-06 ~17:15
- **Completed**: 2026-06-07 ~00:00
- **Total time**: ~6h 45m
- **Status**: SUCCESS — crash fix confirmed (siglip2 R2 fold 4 passed)
- **R1 Results**: dinov2=85.08%, dinov2_large=88.23%, siglip2=94.35%
- **R2 Results**: dinov2=85.91% (+0.83%), dinov2_large=89.16% (+0.93%), siglip2=94.35% (0.00%)
- **Key finding**: min=30 worse than min=20 across all models — too much repetition overfits. Sweet spot is min=20.
- **Key finding**: crash fix (del train_fold/train_fold_raw/val_fold) confirmed — fold 4 passed cleanly.

### run_004 — pipeline.py (SigLIP2-only pseudo + upsampling min=20)
- **Started**: 2026-06-06 ~11:28
- **Completed**: 2026-06-06 ~17:10
- **Total time**: ~5h 42m (crashed in R2)
- **Status**: CRASHED — siglip2 R2 fold 4 silent OOM
- **Crash location**: upsampled train_fold_raw/train_fold/val_fold not deleted between folds → cumulative RAM pressure → virtual address space exhaustion at fold 4
- **R1 Results**: dinov2=85.91% (+0.92%), dinov2_large=88.41% (+0.83%), siglip2=95.18% (+0.37%) ← new CV best
- **R2 partial**: dinov2=86.28% (+0.37%), dinov2_large=88.97% (+0.56%), siglip2 R2 never completed
- **Key finding**: upsampling min=20 helps all models; SigLIP2-only pseudo-labels confirmed better than blended
- **Fix applied**: del train_fold/train_fold_raw/val_fold per fold → run_004b at min=30
- **Submissions saved**: submission_siglip2_so400m.csv (R1 95.18%), submission_pseudo_dinov2.csv, submission_pseudo_dinov2_large.csv

### run_003 — pipeline.py (DINOv2 + DINOv2-Large + SigLIP2, 3-model ensemble + pseudo R2)
- **Started**: 2026-06-06 ~03:15
- **Completed**: 2026-06-06 ~12:45
- **Total time**: ~9h 30m
- **Status**: SUCCESS (first complete run)
- **R1 Results**: dinov2=84.99%, dinov2_large=87.58%, siglip2_so400m=94.81%, routed_ensemble=93.70%
- **R2 Results**: dinov2=84.71% (−0.28%), dinov2_large=87.86% (+0.28%), siglip2_so400m=94.81% (0.00%)
- **Key findings**:
  - Routed ensemble (93.70%) is 1.11% BELOW SigLIP2 alone — class routing counterproductive when mixing very different quality models
  - Blended pseudo-labels (3-model average) had near-zero effect: DINOv2 −0.28%, DINOv2-Large +0.28%, SigLIP2 unchanged
  - SigLIP2 leads 45/100 classes, DINOv2 leads 36/100 (all easy, F1=1.0), DINOv2-Large 10/100, routed 9/100
  - Best submission: submission_siglip2_so400m.csv (R1, 94.81% CV → 94.545% Kaggle)
- **Fixes applied**: transform resolution mismatch, fold model CPU RAM OOM (store_fold_models=False in R2)
- **Next actions**: run_004 — SigLIP2-only pseudo-labels + upsampling

### run_001 — pipeline.py (DINOv2 + DINOv2-Large + SigLIP2, first attempt)
- **Started**: 2026-06-06 (overnight)
- **Completed**: 2026-06-06 ~02:49
- **Total time**: unknown (crashed)
- **Status**: CRASHED — pseudo-label step (transform resolution mismatch)
- **Crash location**: `get_test_probs` called with DINOv2's 224px transform for all models; SigLIP2 expects 378px (729 patches, got 324)
- **R1 Results**: dinov2=84.99%, dinov2_large=87.49%, siglip2=94.81%, routed_ensemble=TBD (submission_routed.csv saved)
- **Kaggle**: not submitted
- **Next actions applied**: fixed transform mismatch (per-model transforms in pseudo-label loop) → run_002

### run_002 — pipeline.py (retry with transform fix)
- **Started**: 2026-06-06 ~02:49
- **Completed**: 2026-06-06 ~07:30
- **Total time**: ~4h 41m (crashed)
- **Status**: CRASHED — SigLIP2 R2 fold 2 silent OOM
- **Crash location**: fold models from DINOv2 (5×~600MB) + DINOv2-Large (5×~1.2GB) held in CPU RAM while SigLIP2 R2 fold 2 tried to mmap 1.6GB from disk → virtual address space exhaustion
- **R1 Results**: dinov2=84.99%, dinov2_large=87.67%, siglip2=94.81%
- **R2 partial**: dinov2=84.71% (−0.28%), dinov2_large fold 1=87.96%, SigLIP2 R2 never completed
- **Kaggle**: not submitted
- **Next actions applied**: store_fold_models=False in R2 (free models after each model's test CSV, before next model loads) → run_003
