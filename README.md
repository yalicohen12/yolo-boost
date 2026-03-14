<div align="center">

# yolo-boost

**Automated hyperparameter optimization for YOLO object detection — smarter than grid search, faster than guessing.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange?logo=mlflow)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/search-Optuna%20TPE-blueviolet)](https://optuna.org/)
[![Ultralytics](https://img.shields.io/badge/model-YOLO-black?logo=ultralytics)](https://docs.ultralytics.com/)

</div>

---

`yolo-boost` wraps [Ultralytics YOLO](https://docs.ultralytics.com/) in an [Optuna](https://optuna.org/) TPE search loop.
Instead of manually tuning hyperparameters, you run one command and let Bayesian optimization find the best configuration for your object detection dataset — with every trial tracked in MLflow.

```
yolo-boost run --preset accuracy --data data.yaml --device 0
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/932d46b8-1201-4c69-be7b-69779cec6766" alt="yolo-boost demo" width="900"/>
</div>

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Install as a Python Package](#install-as-a-python-package)
- [Docker](#docker)
- [Your First Run](#your-first-run)
- [Presets](#presets)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [MLflow Dashboard](#mlflow-dashboard)
- [Team Collaboration](#team-collaboration)
- [Output Structure](#output-structure)
- [Offline Usage](#offline-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features

| | |
|---|---|
| **Bayesian search (TPE)** | Optuna's Tree-structured Parzen Estimator — not random or grid search |
| **Object detection focused** | Built around YOLO's training pipeline, searches across model sizes (n / s / m / l / x) |
| **Trial pruning** | `MedianPruner` kills underperforming trials early to save GPU time |
| **Focused search mode** | Narrow the search to the highest-impact params for faster TPE convergence |
| **Multiple optimization targets** | mAP50-95, mAP50, speed, balanced, precision, recall |
| **Configuration presets** | One-flag quick-start templates for the most common use cases |
| **Rich MLflow tracking** | Per-epoch curves, study progression, Optuna HTML plots, system metrics |
| **Auto image size** | Reads your dataset and picks the right `imgsz` — not a tuned param |
| **Rich terminal output** | Color-coded panels, trial summaries, progress bars via `rich` |
| **Resumable studies** | SQLite / PostgreSQL storage lets you continue interrupted runs |

---

## How It Works

```
  Trial 1–5 (exploration)          Trial 6+ (exploitation)
  ───────────────────────          ────────────────────────────────────────
  Near-random sampling to     →    TPE splits history into "good" (top 25%)
  collect baseline data            and "bad" trials, fits a probability
                                   model, and proposes params where
                                   p(good) / p(bad) is highest.
```

After only **20–50 trials**, TPE typically finds configurations that would require hundreds of random trials to stumble upon. A `MedianPruner` watches every epoch and kills trials that fall behind the running median — saving GPU hours without sacrificing quality.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | 3.11 recommended |
| PyTorch | Install separately for your CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/) |
| NVIDIA GPU | Highly recommended; CPU training works but is slow |
| Docker + Docker Compose | Only needed for the Docker path |
| NVIDIA Container Toolkit | Only needed if you want GPU access inside Docker |

---

## Install as a Python Package

### From GitLab (recommended)

```bash
pip install git+https://github.com/yalicohen12/yolo-boost.git
```

That's it. The `yolo-boost` command is now in your PATH.

### Local / development install

```bash
git clone https://github.com/yalicohen12/yolo-boost.git
cd yolo-boost
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

The `-e` flag (editable) means any changes you make to the source take effect immediately without reinstalling.

### Verify

```bash
yolo-boost --help
```

---

## Docker

Prefer a fully isolated environment with MLflow pre-wired? A `Dockerfile` and `docker-compose.yml` are included.

### 1. Build

```bash
git clone https://github.com/yalicohen12/yolo-boost.git
cd yolo-boost
docker compose build
```

### 2. Start MLflow

```bash
docker compose up mlflow -d
# Open http://localhost:5000
```

### 3. Scaffold config and run

```bash
docker compose run boost init

# Mount your dataset into ./data/ first, then:
docker compose run boost run --preset quick --data /workspace/data/data.yaml
```

> **GPU inside Docker** — the `docker-compose.yml` includes the NVIDIA deploy block. Comment it out and set `DEVICE=cpu` if you're on CPU only.

---

## Your First Run

### 1. Scaffold config

```bash
yolo-boost init
# Edit .yolo-boost-config — set DATA_YAML to your dataset path
```

### 2. Start MLflow (in a separate terminal)

```bash
mlflow ui --backend-store-uri ./mlruns --serve-artifacts --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

### 3. Run

```bash
# Quick sanity check — 5 trials, 10 epochs, small models only
yolo-boost run --preset quick --data data.yaml

# GPU, full accuracy search
yolo-boost run --preset accuracy --data data.yaml --device 0

# Production run persisted to SQLite (resumable)
yolo-boost run --preset production --data data.yaml --device 0 \
  --storage sqlite:///optuna.db
```

### 4. Grab the best weights

```bash
cat best_hyperparameters.yaml          # full config
ls runs/optuna/*/trial_*/weights/best.pt
```

---

## Presets

```bash
yolo-boost run --list-presets
```

| Preset | Trials | Epochs | Models | Best For |
|--------|--------|--------|--------|----------|
| **quick** | 5 | 10 | yolo11n/s, yolo12n/s | Testing setup, fast iteration |
| **focused** | 30 | 50 | yolo11n/s/m, yolo12n/s/m | Best TPE convergence (key params only) |
| **accuracy** | 30 | 50 | yolo11m/l/x, yolo12m/l/x | Maximum accuracy |
| **speed** | 20 | 30 | yolo11n/s, yolo12n/s | Fast inference / edge deployment |
| **balanced** | 25 | 40 | yolo11n/s/m, yolo12n/s/m | Accuracy + speed tradeoff |
| **production** | 50 | 100 | yolo11n/s/m/l, yolo12n/s/m/l | Final, thorough optimization |

---

## CLI Reference

```
yolo-boost init                   Scaffold .yolo-boost-config in current directory
yolo-boost run [OPTIONS]          Run hyperparameter optimization or baseline training

Run options:
  --preset PRESET                 Use a preset (quick/focused/accuracy/speed/balanced/production)
  --list-presets                  Show all presets and exit
  --data PATH                     Path to data.yaml (default: data.yaml)
  --trials N                      Number of optimization trials
  --epochs N                      Epochs per trial
  --patience N                    Early stopping patience
  --models MODEL [...]            YOLO models to search (e.g. yolo11n.pt yolo11s.pt)
  --device DEVICE                 Training device (cpu, 0, 1, cuda:0 …)
  --optimization-metric METRIC    mAP50-95 | mAP50 | precision | recall | speed | balanced
  --mlflow-uri URI                MLflow tracking URI
  --experiment NAME               MLflow experiment name
  --study-name NAME               Optuna study name
  --storage URI                   Optuna storage for persistence (sqlite:///optuna.db, postgresql://…)
  --run-name NAME                 Custom run name (default: auto timestamp)
  --baseline                      Single default training run — no optimization, just a reference point
  --model MODEL                   Model for baseline run
```

**Priority order:** CLI flags → preset → `.yolo-boost-config` → hardcoded defaults

---

## Configuration

### `.yolo-boost-config`

Generate a documented template:

```bash
yolo-boost init
```

Key settings:

```bash
# ── Tracking ──────────────────────────────────
MLFLOW_TRACKING_URI=./mlruns         # or http://your-mlflow-server:5000
MLFLOW_EXPERIMENT_NAME=yolo-optuna-boost

# ── Data & device ─────────────────────────────
DATA_YAML=data.yaml
DEVICE=cpu                           # or 0, 1, cuda:0

# ── Study ─────────────────────────────────────
N_TRIALS=20
EPOCHS=50
PATIENCE=50
OPTIMIZATION_METRIC=mAP50-95        # mAP50 | precision | recall | speed | balanced

# ── Search space ──────────────────────────────
MODEL_VERSIONS=yolo11n.pt,yolo11s.pt,yolo11m.pt,yolo11l.pt
OPTIMIZER_OPTIONS=SGD,Adam,AdamW,NAdam
BATCH_OPTIONS=8,16,32,64

# ── Focused search (leave blank to search all params) ──
SEARCH_PARAMS=lr0,optimizer,batch,box,cls,mosaic
```

### Hyperparameter ranges

All ranges are `min,max` and can be overridden:

```bash
# Learning rate
LR0_RANGE=1e-5,1e-1
LRF_RANGE=0.01,1.0
MOMENTUM_RANGE=0.8,0.99
WEIGHT_DECAY_RANGE=0.0,0.01

# Loss weights
BOX_RANGE=1.0,20.0
CLS_RANGE=0.1,4.0
DFL_RANGE=0.5,4.0

# Augmentation
DEGREES_RANGE=0.0,45.0
MOSAIC_RANGE=0.0,1.0
ERASING_RANGE=0.0,0.9
# … see .yolo-boost-config for the full list
```

---

## MLflow Dashboard

Every trial — parameters, metrics, weights, and Optuna plots — is tracked automatically.

```bash
mlflow ui \
  --backend-store-uri ./mlruns \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Open **http://localhost:5000**.

> `--serve-artifacts` is required for artifact uploads (weights, plots) to work. Without it they are skipped silently.
> `--backend-store-uri` must match `MLFLOW_TRACKING_URI` in your `.yolo-boost-config`. Both default to `./mlruns`.

```
Experiment: yolo-optuna-boost
└── Parent run: yolo-optimization_run_20260311_120000
    ├── trial_mAP50_95 time-series (study progression)
    ├── Optuna HTML plots (optimization_history, param_importances …)
    ├── System metrics (CPU, RAM, GPU)
    ├── trial_0/                    ← nested child runs
    │   ├── Per-epoch curves (mAP, losses)
    │   ├── YOLO plots (confusion matrix, PR curve …)
    │   └── best.pt weights
    ├── trial_1/
    └── …
```

---

## Team Collaboration

### Shared Optuna study

```bash
# Everyone points at the same DB — TPE learns from all contributors
yolo-boost run \
  --storage postgresql://user:pass@host:5432/optuna_db \
  --study-name shared-study \
  --data data.yaml
```

### Shared MLflow server

```bash
# Server (one machine)
mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts

# Each workstation's .yolo-boost-config
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

> Without `--serve-artifacts`, artifact uploads (weights, plots) are skipped with a warning. Metrics and params always track fine.

---

## Output Structure

```
runs/optuna/
└── run_20260311_120000/            # or your --run-name
    ├── trial_0/
    │   └── weights/
    │       ├── best.pt
    │       └── last.pt
    ├── trial_1/
    └── …
best_hyperparameters.yaml           # auto-generated after each study
```

---

## Offline Usage

By default, Ultralytics downloads model weights the first time you reference them (e.g. `yolo11n.pt`). To run fully offline, download the weights once and point yolo-boost at the local files.

### 1. Download weights once (while online)

```bash
# Let Ultralytics pull them into your CWD
yolo predict model=yolo11n.pt source=. 2>/dev/null || true
yolo predict model=yolo11s.pt source=. 2>/dev/null || true
# repeat for any model you need
```

Or just copy `.pt` files you already have from elsewhere.

### 2. Run offline

**Option A — place `.pt` files in your working directory:**

```bash
ls .
# yolo11n.pt  yolo11s.pt  data.yaml  .yolo-boost-config

yolo-boost run --preset quick --data data.yaml
# Ultralytics finds them in CWD — no download
```

**Option B — pass full paths (recommended for a shared weights folder):**

```bash
yolo-boost run \
  --models /shared/weights/yolo11n.pt /shared/weights/yolo11s.pt \
  --data data.yaml
```

Or set it permanently in `.yolo-boost-config`:

```bash
MODEL_VERSIONS=/shared/weights/yolo11n.pt,/shared/weights/yolo11s.pt
```

> Ultralytics only attempts a download when the model name has no path component and the file isn't found in the CWD. A full path always loads locally.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **Too slow** | Use `--preset quick` or reduce `--epochs` |
| **Out of memory** | Set `BATCH_OPTIONS=4,8` in `.yolo-boost-config` |
| **Models not converging** | Increase `EPOCHS`, widen `LR0_RANGE` |
| **Stale MLflow runs** | Handled automatically on startup; or run `mlflow gc` |
| **Interrupted study** | Re-run with the same `--storage` and `--study-name` to resume |
| **Images not found** | Set `path` in your `data.yaml` to an absolute path — Ultralytics resolves relative paths from its own `datasets_dir`, not from where your `data.yaml` lives |
| **Models downloading every run** | Place `.pt` files in your CWD, or pass full paths: `--models /path/to/yolo11n.pt` (or set `MODEL_VERSIONS=/path/to/yolo11n.pt` in `.yolo-boost-config`). Ultralytics only downloads if the file isn't found locally |

---

## Contributing

1. Fork the repo and create a feature branch
2. Test your changes with `yolo-boost run --preset quick --data data.yaml`
3. Open a merge request — describe what you changed and why

Potential areas for contribution:
- Parallel GPU trials (`n_jobs > 1`)
- ONNX / TensorRT auto-export of the best model
- W&B integration as an alternative to MLflow
- Slack / Discord webhook notifications on study completion
- `yolo-boost compare` command for side-by-side study comparison
- Multi-objective optimization (accuracy vs. latency Pareto front)

---

## License

MIT — see [LICENSE](LICENSE).
