# CSE 144 Final Project — Claude Agent Guide

## Project overview
100-class image classification (~1079 train images). RTX 5070 Ti, WSL2, Python 3.12.
Key scripts: `search.py` (hyperparameter sweep), `pipeline.py` (full CV run + Kaggle submission).

## Environment
**Always use Windows Python** for training — native CUDA, no WSL filesystem overhead.
```
Windows Python: /mnt/c/Users/Caleb Cho/code/school/cse144-final/.venv-win/Scripts/python.exe
WSL Python:     /mnt/c/Users/Caleb Cho/code/school/cse144-final/.venv/bin/python3  (do NOT use for training)
```
Working directory: `/mnt/c/Users/Caleb Cho/code/school/cse144-final`

## Running with live monitoring

**IMPORTANT — always use a subagent to launch scripts.** Never run search.py or pipeline.py blocking the main chat. Spawn an Agent, have it start the process, report the PID back, then arm the Monitor in the main chat.

### Standard launch sequence (agent does steps 1-2, main chat does step 3)

**Step 1 — subagent starts the process (Windows Python, always):**
```bash
cd "/mnt/c/Users/Caleb Cho/code/school/cse144-final" && .venv-win/Scripts/python.exe -u search.py 2>&1 | tee search_run.log &
BGPID=$! && echo "PID=$BGPID" && sleep 5 && head -5 search_run.log
```
Subagent reports the PID back to main chat. User needs the PID to kill the run if needed (multiple runs may be active simultaneously).

**Step 2 — subagent confirms log is live** (wait until search_run.log is non-empty).

**Step 3 — main chat arms the Monitor:**
```
Monitor(
  command="tail -n +1 -f search_run.log | grep --line-buffered '##SIGNAL##\\|Traceback\\|RuntimeError\\|OOM\\|CUDA out\\|✗\\|Done\\. Results'",
  persistent=True
)
```

### Killing a run
**Always kill via a subagent** (keeps main chat free). Use SIGKILL — PyTorch ignores SIGTERM mid-training.
Subagent runs:
```bash
kill -9 <PID> && sleep 1 && pgrep -a python | grep search || echo "stopped"
```
Multiple concurrent runs write to different log files (e.g. search_run.log, pipeline_run.log).

### pipeline.py
Same pattern — subagent runs (Windows Python):
```bash
cd "/mnt/c/Users/Caleb Cho/code/school/cse144-final" && .venv-win/Scripts/python.exe -u pipeline.py 2>&1 | tee pipeline_run.log &
BGPID=$! && echo "PID=$BGPID" && sleep 5 && head -5 pipeline_run.log
```

## ##SIGNAL## protocol
Both scripts emit structured lines: `##SIGNAL## <event> key=val ...`

| Event | Trigger | Push to phone? |
|---|---|---|
| `config_start` | search config begins | no |
| `fold_start` | fold begins | no |
| `fold_done` | fold complete (acc, f1, train_sec) | no |
| `config_done` | config CV complete (acc, f1) | **yes** |
| `error` | config/model failed | **yes** |
| `search_done` | all configs finished | **yes** |
| `model_start` | pipeline model begins | no |
| `model_done` | pipeline model CV complete | **yes** |
| `pipeline_done` | all pipeline models done (best acc) | **yes** |

When monitoring, send a PushNotification for **config_done**, **model_done**, **error**, **search_done**, **pipeline_done**.
Skip fold-level signals (too noisy for phone).

### PushNotification message formats
- `config_done`: `"[search] cfg {config} {model} — acc={acc} f1={f1}"`
- `model_done`: `"[pipeline] {model} done — acc={acc} f1={f1} ({total_min}m)"`
- `error`: `"[ERROR] cfg {config} {model}: {msg}"`
- `search_done`: `"[search] DONE — {total} configs, results in {results}"`
- `pipeline_done`: `"[pipeline] DONE — best={best} acc={acc}"`

## Key config
Edit `config.py` to change models/hyperparameters before running pipeline.
Edit `CONFIGS_TO_RUN` set in `search.py` to select which configs to run.

## Kaggle submission
```bash
python3 pipeline.py --submit --message "run description"
```
Competition: `ucsc-cse-144-spring-2026-final-project`

## Current best
DINOv2-base: ~84.24% val accuracy (config 1: unfreeze=2, lr=5e-5, no aug).
DINOv2-Large (facebook/dinov2-large): configs 56-63 in search.py, currently running.
