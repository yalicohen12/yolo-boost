# YOLO Optuna Boost

Automated hyperparameter optimization for YOLO models using Optuna and MLflow tracking.

## Features

- **Smart Hyperparameter Search**: Uses Optuna's TPE algorithm (not random/brute-force)
- **Multiple Optimization Targets**: Optimize for accuracy (mAP50-95), speed, or balanced performance
- **Configuration Presets**: Quick-start templates for common use cases
- **MLflow Integration**: Track all experiments with metrics, parameters, and artifacts
- **Flexible Configuration**: Use presets, .env files, or command-line arguments
- **Model Size Optimization**: Automatically finds the best YOLO model variant (n/s/m/l/x)

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install ultralytics optuna mlflow python-dotenv
```

### 2. Prepare Your Data

Create a `data.yaml` file for your dataset:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class1
  1: class2
```

### 3. Configure Settings

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` to customize your setup (optional - presets work out of the box).

### 4. Run Optimization

**Using Presets (Recommended):**

```bash
# Quick experiment (5 trials, 10 epochs, fast models)
python run_study.py --preset quick --data data.yaml

# Accuracy-focused (30 trials, larger models)
python run_study.py --preset accuracy --data data.yaml --device 0

# Speed-focused (optimize for fast inference)
python run_study.py --preset speed --data data.yaml

# Balanced (good accuracy + reasonable speed)
python run_study.py --preset balanced --data data.yaml

# Production (thorough 50-trial search)
python run_study.py --preset production --data data.yaml --device 0
```

**Custom Configuration:**

```bash
python run_study.py \
  --data data.yaml \
  --trials 20 \
  --device 0 \
  --models yolo11n.pt yolo11s.pt yolo11m.pt \
  --optimization-metric mAP50-95
```

### 5. View Results

Start MLflow UI:

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 in your browser to:
- Compare trial results
- View hyperparameter importance
- Download best model weights
- Analyze metrics over time

## MLflow Organization

Runs are automatically grouped by **study name** in MLflow experiments:

```bash
# Creates experiment "accuracy-study"
python run_study.py --study-name "accuracy-study" --preset accuracy --data data.yaml

# Creates experiment "speed-study"
python run_study.py --study-name "speed-study" --preset speed --data data.yaml
```

**MLflow UI Structure:**
```
📁 Experiment: "accuracy-study"
   └── Parent Run: "accuracy-study_run_20251221_143022"
       ├── trial_0
       ├── trial_1
       └── trial_2

📁 Experiment: "speed-study"
   └── Parent Run: "speed-study_run_20251221_153045"
       ├── trial_0
       └── trial_1
```

**Custom experiment names:**
```bash
# Override: use custom experiment name
python run_study.py --study-name "my-study" --experiment "custom-name" --data data.yaml
```

## Available Presets

List all presets with details:

```bash
python run_study.py --list-presets
```

| Preset | Trials | Epochs | Models | Best For |
|--------|--------|--------|--------|----------|
| **quick** | 5 | 10 | n, s | Quick testing, iteration |
| **accuracy** | 30 | 50 | m, l, x | Maximum accuracy |
| **speed** | 20 | 30 | n, s | Fast inference |
| **balanced** | 25 | 40 | n, s, m | Production deployment |
| **production** | 50 | 100 | n, s, m, l | Final optimization |

## Optimization Metrics

Control what to optimize via `OPTIMIZATION_METRIC` in `.env` or `--optimization-metric` flag:

- **mAP50-95** (default): Best overall accuracy, standard COCO metric
- **mAP50**: Faster convergence, good for experimentation
- **speed**: Optimizes for fast inference (balances accuracy + model size)
- **balanced**: 70% accuracy + 30% speed
- **precision**: Minimize false positives
- **recall**: Minimize false negatives

Example:

```bash
python run_study.py --preset quick --optimization-metric speed
```

## Configuration

### Priority Order

1. Command-line arguments (highest)
2. Preset values
3. .env file
4. Hardcoded defaults (lowest)

### .env Configuration

Key settings in `.env`:

```bash
# Data
DATA_YAML=data.yaml

# Device
DEVICE=cpu  # or 0, 1, 2 for GPU

# Optimization
OPTIMIZATION_METRIC=mAP50-95
MODEL_VERSIONS=yolo11n.pt,yolo11s.pt,yolo11m.pt,yolo11l.pt

# Training
EPOCHS_RANGE=10,10  # Fixed 10 epochs (or 10,50 to optimize)
BATCH_OPTIONS=8,16,32,64
# IMGSZ_OPTIONS=320,416,512,640  # Commented = use defaults

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=yolo-optuna-boost

# Optuna
OPTUNA_STUDY_NAME=yolo-optimization
OPTUNA_N_TRIALS=20
OPTUNA_STORAGE=  # Optional: sqlite:///optuna.db for persistence
```

### Hyperparameter Ranges

Fine-tune search ranges in `.env`:

```bash
# Learning Rate
LR0_RANGE=1e-5,1e-1        # Initial learning rate
LRF_RANGE=0.01,1.0         # Final learning rate fraction

# Augmentation
HSV_H_RANGE=0.0,0.1        # Hue augmentation
DEGREES_RANGE=0.0,45.0     # Rotation degrees
MOSAIC_RANGE=0.0,1.0       # Mosaic augmentation probability

# Loss Weights
BOX_RANGE=0.02,0.2         # Box loss weight
CLS_RANGE=0.2,4.0          # Classification loss weight
```

See `.env.example` for all available parameters.

## Command-Line Reference

```bash
python run_study.py [OPTIONS]

Options:
  --preset PRESET              Use preset (quick/accuracy/speed/balanced/production)
  --list-presets              Show all presets
  --data PATH                 Path to data.yaml (required)
  --trials N                  Number of optimization trials
  --epochs N                  Epochs per trial
  --models MODEL [MODEL ...]  YOLO models to try (e.g., yolo11n.pt yolo11s.pt)
  --device DEVICE             Training device (cpu, 0, 1, cuda:0, etc.)
  --optimization-metric M     What to optimize (mAP50-95/mAP50/speed/balanced)
  --mlflow-uri URI           MLflow tracking URI
  --experiment NAME          MLflow experiment name
  --study-name NAME          Optuna study name
  --storage URI              Optuna storage (sqlite:///optuna.db for persistence)
```

## How It Works

### Optuna TPE Algorithm

YOLO Optuna Boost uses **Tree-structured Parzen Estimator (TPE)**, not random search:

1. **Trials 0-5**: Initial exploration with semi-random parameters
2. **Trial 6+**: Smart optimization
   - Analyzes all previous trials
   - Builds probability model of good vs. bad parameter regions
   - Suggests parameters likely to improve performance
   - Balances exploration (new areas) vs. exploitation (promising areas)

**Example learning progression:**

```
Trial 0:  lr0=0.001, yolo11s → mAP50-95: 0.65
Trial 1:  lr0=0.01, yolo11m → mAP50-95: 0.72  ✓ Better!
Trial 2:  lr0=0.008, yolo11m → mAP50-95: 0.75  ✓ Learning: lr ~0.01 + yolo11m works
Trial 3:  lr0=0.009, yolo11m → mAP50-95: 0.78  ✓ Converging...
```

This converges to optimal parameters in **20-50 trials** vs. thousands for brute-force.

### What Gets Optimized

**Automatically tuned:**
- Model size (n/s/m/l/x variants)
- Learning rates (lr0, lrf, momentum)
- Batch size and image size
- Data augmentation (HSV, rotation, mosaic, mixup, etc.)
- Loss weights (box, cls, dfl)

**Tracked in MLflow:**
- mAP50, mAP50-95
- Precision, Recall
- Speed score (model size)
- All hyperparameters
- Best model weights

## Examples

### Example 1: Quick Test on CPU

```bash
python run_study.py --preset quick --data my_data.yaml
```

Runs 5 trials with small models on CPU (10 epochs each) - great for testing setup.

### Example 2: Production Optimization on GPU

```bash
python run_study.py \
  --preset production \
  --data data.yaml \
  --device 0 \
  --experiment "production-detector-v1" \
  --storage sqlite:///optuna.db
```

Thorough 50-trial search on GPU 0, persists results to database.

### Example 3: Speed-Optimized for Edge Deployment

```bash
python run_study.py \
  --preset speed \
  --data data.yaml \
  --device 0 \
  --optimization-metric speed
```

Finds fastest model with acceptable accuracy for edge devices.

### Example 4: Custom Configuration

```bash
python run_study.py \
  --data data.yaml \
  --trials 15 \
  --epochs 25 \
  --models yolo11n.pt yolo11s.pt \
  --device cuda:0 \
  --optimization-metric balanced
```

Custom setup: 15 trials, 25 epochs, only small models, balanced optimization.

### Example 5: Baseline Training (No Optimization)

Run a single training with default hyperparameters for comparison:

```bash
# Single baseline run
python run_study.py \
  --baseline \
  --data data.yaml \
  --epochs 50 \
  --device 0

# Baseline with specific model
python run_study.py \
  --baseline \
  --model yolo11m.pt \
  --data data.yaml \
  --study-name "my-optimization"
```

This creates a baseline run in the same MLflow experiment for comparison with optimized trials.

## Team Collaboration

### Shared Database

Use PostgreSQL for team-wide study sharing:

```bash
# In .env
OPTUNA_STORAGE=postgresql://user:password@host:5432/optuna_db

# Or command-line
python run_study.py --storage postgresql://...
```

All team members see the same study results and Optuna learns from everyone's trials.

### MLflow Server

Run MLflow on a shared server:

```bash
# Server
mlflow server --host 0.0.0.0 --port 5000

# Team members in .env
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## Best Practices

1. **Start with presets**: Use `--preset quick` to test, then `--preset production` for final
2. **Use GPU when available**: `--device 0` for 10-20x speedup
3. **Persist studies**: Use `--storage sqlite:///optuna.db` to resume interrupted runs
4. **Monitor in MLflow**: Keep MLflow UI open to watch progress
5. **Adjust ranges**: If results plateau, tighten hyperparameter ranges in `.env`
6. **Choose right metric**:
   - Research/competition: `mAP50-95`
   - Quick iteration: `mAP50`
   - Production deployment: `balanced` or `speed`

## Troubleshooting

### Issue: Training is too slow

**Solution**: Use faster preset or reduce epochs
```bash
python run_study.py --preset quick --epochs 5
```

### Issue: Models not converging

**Solution**: Increase epochs or adjust learning rate ranges
```bash
# In .env
EPOCHS_RANGE=30,50
LR0_RANGE=1e-4,1e-2
```

### Issue: MLflow run conflicts

**Solution**: Already handled automatically - script clears stale runs

### Issue: Out of memory

**Solution**: Reduce batch size or image size
```bash
# In .env
BATCH_OPTIONS=4,8
IMGSZ_OPTIONS=320,416
```

## Run Organization

Each optimization study gets its own **timestamped folder** to prevent overwrites:

```
runs/optuna/
├── run_20251221_143022/    # Auto-generated timestamp
│   ├── trial_0/
│   │   ├── weights/
│   │   │   └── best.pt
│   │   └── ...
│   ├── trial_1/
│   └── trial_2/
├── run_20251221_153045/    # Another study run
│   └── ...
└── production-v1/          # Custom run name
    └── ...
```

### Custom Run Names

Use meaningful names for important experiments:

```bash
# Auto-generated timestamp (default)
python run_study.py --preset quick --data data.yaml
# Saves to: runs/optuna/run_20251221_143022/

# Custom name for production experiments
python run_study.py --preset production --run-name "production-v1" --data data.yaml
# Saves to: runs/optuna/production-v1/

# Team collaboration - use your name
python run_study.py --preset balanced --run-name "alice-experiment-1" --data data.yaml
# Saves to: runs/optuna/alice-experiment-1/
```

**Benefits:**
- ✅ No overwrites or confusion
- ✅ Easy to track when experiments ran
- ✅ Meaningful names for important runs
- ✅ Safe to re-run studies anytime

## File Structure

```
yolo-boost/
├── README.md                     # This file
├── CODEBASE.md                   # Detailed code documentation
├── .env.example                  # Example configuration
├── .env                          # Your configuration (git-ignored)
├── run_study.py                  # Main entry point
├── train_yolo_optuna.py          # Core optimization logic
├── presets.py                    # Preset configurations
├── data.yaml                     # Your dataset config
├── runs/optuna/                  # Training outputs
│   ├── run_TIMESTAMP/           # Each study in its own folder
│   │   ├── trial_0/
│   │   ├── trial_1/
│   │   └── ...
│   └── ...
└── best_hyperparameters.yaml     # Best params (auto-generated)
```

## Contributing

Suggestions for new presets or features? Open an issue or PR!

## License

MIT License - use freely for your projects.

---

**Questions?** Check the Optuna docs: https://optuna.org
