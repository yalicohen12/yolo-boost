# YOLO Boost

Automated hyperparameter optimization for YOLO models using Optuna (TPE) and MLflow tracking.

## Features

- **Smart Search (TPE)** — Optuna's Tree-structured Parzen Estimator, not random/grid search
- **YOLO11 & YOLO12** — searches across model families and sizes (n/s/m/l/x)
- **Trial Pruning** — MedianPruner kills underperforming trials early to save GPU time
- **Focused Search Mode** — reduce to 10 high-impact params for faster TPE convergence
- **Multiple Optimization Targets** — mAP50-95, mAP50, speed, balanced, precision, recall
- **Configuration Presets** — quick-start templates for common use cases
- **Rich MLflow Tracking** — per-epoch curves, study progression, Optuna HTML plots, system metrics
- **Auto Image Size** — reads your dataset and picks the right `imgsz` automatically
- **Rich Terminal Output** — color-coded panels, trial summaries, and progress via `rich`

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Scaffold config

```bash
yolo-boost init
mv .env.yolo-boost.example .env.yolo-boost
# Edit .env.yolo-boost with your settings
```

### 3. Run optimization

```bash
# Quick test (5 trials, 10 epochs, small models)
yolo-boost run --preset quick --data data.yaml

# GPU, accuracy-focused
yolo-boost run --preset accuracy --data data.yaml --device 0

# Production (50 trials, persisted to DB)
yolo-boost run --preset production --data data.yaml --device 0 --storage sqlite:///optuna.db
```

### 4. View results

```bash
mlflow ui
```

Open http://localhost:5000 to compare trials, view hyperparameter importances, and download best weights.

---

## Presets

```bash
yolo-boost run --list-presets
```

| Preset | Trials | Epochs | Models | Best For |
|--------|--------|--------|--------|----------|
| **quick** | 5 | 10 | yolo11n/s, yolo12n/s | Testing setup, fast iteration |
| **focused** | 30 | 50 | yolo11n/s/m, yolo12n/s/m | Best TPE convergence (10 key params) |
| **accuracy** | 30 | 50 | yolo11m/l/x, yolo12m/l/x | Maximum accuracy |
| **speed** | 20 | 30 | yolo11n/s, yolo12n/s | Fast inference |
| **balanced** | 25 | 40 | yolo11n/s/m, yolo12n/s/m | Accuracy + speed tradeoff |
| **production** | 50 | 100 | yolo11n/s/m/l, yolo12n/s/m/l | Final optimization |

---

## CLI Reference

```
yolo-boost init                   Scaffold .env.yolo-boost.example in current directory
yolo-boost run [OPTIONS]          Run optimization or baseline training

Run options:
  --preset PRESET                 Use a preset (quick/focused/accuracy/speed/balanced/production)
  --list-presets                  Show all presets and exit
  --data PATH                     Path to data.yaml (default: data.yaml)
  --trials N                      Number of optimization trials
  --epochs N                      Epochs per trial
  --patience N                    Early stopping patience
  --models MODEL [...]            YOLO models to search (e.g. yolo11n.pt yolo11s.pt)
  --device DEVICE                 Training device (cpu, 0, 1, cuda:0, ...)
  --optimization-metric METRIC    mAP50-95 | mAP50 | precision | recall | speed | balanced
  --mlflow-uri URI                MLflow tracking URI
  --experiment NAME               MLflow experiment name
  --study-name NAME               Optuna study name
  --storage URI                   Optuna storage for persistence (e.g. sqlite:///optuna.db)
  --run-name NAME                 Custom run name (default: auto timestamp)
  --baseline                      Single default training run (no optimization)
  --model MODEL                   Model for baseline run
```

**Priority order:** CLI flags → preset → `.env.yolo-boost` → hardcoded defaults

---

## Configuration

### .env.yolo-boost

```bash
# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=yolo-optuna-boost

# Data & device
DATA_YAML=data.yaml
DEVICE=cpu  # or 0, 1, cuda:0

# Training (fixed per run — not Optuna search params)
N_TRIALS=20
EPOCHS=50
PATIENCE=50

# Optimization target
OPTIMIZATION_METRIC=mAP50-95

# Search space
MODEL_VERSIONS=yolo11n.pt,yolo11s.pt,yolo11m.pt,yolo11l.pt
OPTIMIZER_OPTIONS=SGD,Adam,AdamW,NAdam
BATCH_OPTIONS=8,16,32,64

# Focused search: comma-separated params to search (blank = search all 28)
SEARCH_PARAMS=
```

### Hyperparameter ranges

All ranges are `min,max` and can be overridden in `.env.yolo-boost`:

```bash
# Learning rate
LR0_RANGE=1e-5,1e-1
LRF_RANGE=0.01,1.0
MOMENTUM_RANGE=0.8,0.99
WEIGHT_DECAY_RANGE=0.0,0.01

# Loss weights (YOLO11/12 defaults: box=7.5, cls=0.5, dfl=1.5)
BOX_RANGE=1.0,20.0
CLS_RANGE=0.1,4.0
DFL_RANGE=0.5,4.0

# Augmentation
DEGREES_RANGE=0.0,45.0
MOSAIC_RANGE=0.0,1.0
ERASING_RANGE=0.0,0.9
# ... see .env.yolo-boost.example for all 28 params
```

---

## How It Works

### TPE (Tree-structured Parzen Estimator)

Trials 1–5 are near-random to collect baseline data. From trial 6 onward, TPE:

1. Splits all completed trials into **good** (top ~25%) and **bad** (the rest)
2. Fits a probability distribution to each group per parameter
3. Proposes values where **p(good) / p(bad)** is highest — regions that produced good results without producing bad ones

This converges to strong hyperparameters in 20–50 trials instead of thousands.

### Pruning

`MedianPruner` reports mAP50-95 to Optuna after every epoch. If a trial is tracking below the median of all previous trials at the same epoch (after a 10-epoch warmup), it's killed early. Saved compute is reported at the end.

### Focused Search

With 28 parameters, TPE needs many trials for good coverage. The `focused` preset and `SEARCH_PARAMS` env var let you narrow the search to the highest-impact parameters (e.g. `lr0, optimizer, batch, box, cls, mosaic`). Everything else holds its YOLO11/12 default.

### What Gets Optimized

28 parameters across 5 categories:
- **Model & training**: model variant, optimizer, batch size
- **Learning rate**: lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, warmup_bias_lr
- **Loss weights**: box, cls, dfl, label_smoothing
- **Color augmentation**: hsv_h, hsv_s, hsv_v, bgr
- **Geometric & advanced augmentation**: degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste, erasing, close_mosaic

Image size is auto-detected from your dataset — not tuned.

---

## MLflow Structure

```
Experiment: yolo-optuna-boost
└── Parent run: yolo-optimization_run_20260311_120000
    ├── Per-trial params and final metrics
    ├── trial_mAP50_95 and best_so_far time-series (study progression)
    ├── Optuna HTML plots (optimization_history, param_importances, ...)
    ├── System metrics (CPU, RAM, GPU)
    ├── trial_0/   ← nested child runs
    │   ├── Per-epoch curves (mAP, losses)
    │   ├── YOLO plots (confusion matrix, PR curve, ...)
    │   └── best.pt weights
    ├── trial_1/
    └── ...
```

---

## Examples

### Baseline comparison

```bash
# Run default YOLO training (no optimization) for a reference point
yolo-boost run --baseline --model yolo11m.pt --data data.yaml --epochs 50 --device 0
```

### Resume interrupted study

```bash
# SQLite storage lets you resume if interrupted
yolo-boost run --preset production --data data.yaml --storage sqlite:///optuna.db
```

### Speed-optimized for edge deployment

```bash
yolo-boost run --preset speed --data data.yaml --device 0 --optimization-metric speed
```

### Custom run

```bash
yolo-boost run \
  --data data.yaml \
  --trials 15 \
  --epochs 25 \
  --models yolo11n.pt yolo11s.pt \
  --device cuda:0 \
  --run-name "experiment-v1"
```

---

## Output Structure

```
runs/optuna/
└── run_20260311_120000/     # or your --run-name
    ├── trial_0/
    │   └── weights/best.pt
    ├── trial_1/
    └── ...
best_hyperparameters.yaml    # auto-generated after each study
```

---

## Team Collaboration

**Shared Optuna DB:**
```bash
yolo-boost run --storage postgresql://user:pass@host:5432/optuna_db --data data.yaml
```
All members contribute to the same study; TPE learns from everyone's trials.

**Shared MLflow server:**
```bash
# Server
mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts

# .env.yolo-boost on each machine
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

> Without `--serve-artifacts`, artifact uploads (weights, plots) are skipped with a warning. Metrics and params always track.

---

## Troubleshooting

**Too slow** → use `--preset quick` or reduce `--epochs`

**Out of memory** → set `BATCH_OPTIONS=4,8` in `.env.yolo-boost`

**Models not converging** → increase `EPOCHS`, tighten `LR0_RANGE`

**Stale MLflow runs** → handled automatically on startup

---

## License

MIT
