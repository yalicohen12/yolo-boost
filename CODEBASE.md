# Codebase Documentation

Complete technical documentation for the YOLO Optuna Boost codebase.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Descriptions](#file-descriptions)
- [Data Flow](#data-flow)
- [Key Components](#key-components)
- [Configuration System](#configuration-system)
- [Extending the Code](#extending-the-code)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────┐
│   User Input    │
│  (CLI/Preset)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  run_study.py   │ ◄── Preset configs (presets.py)
│  Entry Point    │ ◄── .env variables
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ YOLOOptunaTrainer    │
│ (train_yolo_optuna)  │
└────────┬─────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌────────┐
│Optuna │  │MLflow  │
│(TPE)  │  │Tracking│
└───┬───┘  └────────┘
    │
    ▼
┌──────────────┐
│   Ultralytics│
│   YOLO Train │
└──────────────┘
```

## File Descriptions

### `run_study.py`

**Purpose**: Main entry point for running optimization studies.

**Key Functions**:
- `parse_args()`: Parses command-line arguments
- `main()`: Orchestrates preset application, configuration, and study execution

**Flow**:
1. Parse command-line arguments
2. Load preset if specified (`--preset`)
3. Apply preset values (CLI args override preset)
4. Set environment variables for preset batch/image sizes
5. Create `YOLOOptunaTrainer` instance
6. Run optimization study
7. Display results

**Configuration Priority** (highest to lowest):
1. Command-line arguments
2. Preset values
3. .env file
4. Hardcoded defaults

### `train_yolo_optuna.py`

**Purpose**: Core optimization engine integrating Optuna, MLflow, and YOLO.

**Main Class**: `YOLOOptunaTrainer`

#### Constructor (`__init__`)

```python
def __init__(self, data_yaml, mlflow_tracking_uri, experiment_name,
             model_versions, device, optimization_metric, run_name)
```

**Responsibilities**:
- Load configuration from parameters or environment variables
- Generate unique run name (timestamp-based if not provided)
- Parse hyperparameter ranges from .env
- Setup MLflow tracking

**Key Attributes**:
- `self.data_yaml`: Path to YOLO dataset configuration
- `self.device`: Training device (cpu/gpu)
- `self.optimization_metric`: What to optimize (mAP50-95, speed, etc.)
- `self.run_name`: Unique identifier for this study run
- `self.ranges`: Dictionary of hyperparameter search ranges
- `self.model_versions`: List of YOLO models to try

#### Objective Function (`objective`)

```python
def objective(self, trial) -> float
```

**Purpose**: Optuna calls this function for each trial to evaluate hyperparameters.

**Flow**:
1. **Suggest hyperparameters** using Optuna trial
   - Model version (categorical)
   - Learning rates (float, log scale)
   - Batch size (categorical)
   - Image size (categorical)
   - Augmentation parameters (float)
   - Loss weights (float)

2. **Start MLflow run** (nested under experiment)
   - Log all suggested parameters

3. **Train YOLO model**
   - Load suggested model variant
   - Train with suggested hyperparameters
   - Save to unique folder: `runs/optuna/{run_name}/trial_{N}/`

4. **Extract metrics** from training results
   - mAP50, mAP50-95
   - Precision, Recall
   - Calculate speed score based on model size

5. **Log to MLflow**
   - Metrics (mAP, precision, recall, speed)
   - Model artifacts (best.pt weights)

6. **Return optimization metric**
   - Based on `self.optimization_metric`
   - Can be single metric or weighted combination

**Optimization Metric Calculation**:

```python
if metric == 'mAP50-95':
    return map50_95
elif metric == 'speed':
    return map50_95 * 0.5 + model_size_score * 0.5
elif metric == 'balanced':
    return map50_95 * 0.7 + model_size_score * 0.3
# ... etc
```

**Model Size Scoring**:
- Nano (n): 1.0 (fastest)
- Small (s): 0.8
- Medium (m): 0.6
- Large (l): 0.4
- X-Large (x): 0.2 (slowest)

#### Optimize Function (`optimize`)

```python
def optimize(self, n_trials, study_name, storage) -> optuna.Study
```

**Purpose**: Run the optimization study.

**Flow**:
1. **Clear stale MLflow runs**
   - Prevents conflicts from interrupted runs
   - Loops until no active runs remain

2. **Create Optuna study**
   - Direction: 'maximize' (higher metric = better)
   - Can load existing study from storage (resume)

3. **Run optimization**
   - Calls `objective()` for each trial
   - TPE sampler learns from previous trials

4. **Print best results**
   - Best trial number
   - Best metric value
   - Best hyperparameters

5. **Save best params** to YAML file

### `presets.py`

**Purpose**: Predefined configuration sets for common use cases.

**Structure**:

```python
PRESETS = {
    'preset_name': {
        'description': 'Human-readable description',
        'n_trials': int,
        'epochs': int,
        'models': ['model1.pt', 'model2.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': 'comma,separated,values',
        'imgsz_options': 'comma,separated,values'
    }
}
```

**Functions**:
- `get_preset(name)`: Returns preset dict or raises error
- `list_presets()`: Prints all presets with descriptions

**Available Presets**:

| Preset | Use Case | Trials | Epochs | Models |
|--------|----------|--------|--------|--------|
| quick | Fast iteration | 5 | 10 | n, s |
| accuracy | Max accuracy | 30 | 50 | m, l, x |
| speed | Fast inference | 20 | 30 | n, s |
| balanced | Production | 25 | 40 | n, s, m |
| production | Thorough search | 50 | 100 | n, s, m, l |

### `.env` Configuration File

**Purpose**: Centralized configuration for all settings.

**Categories**:

#### MLflow Settings
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=yolo-optuna-boost
```

#### Optuna Settings
```bash
OPTUNA_STUDY_NAME=yolo-optimization
OPTUNA_N_TRIALS=20
OPTUNA_STORAGE=  # Optional: sqlite:///optuna.db
```

#### Training Settings
```bash
DATA_YAML=data.yaml
DEVICE=cpu  # or 0, 1, cuda:0, etc.
OPTIMIZATION_METRIC=mAP50-95
```

#### Model Settings
```bash
MODEL_VERSIONS=yolo11n.pt,yolo11s.pt,yolo11m.pt,yolo11l.pt
```

#### Training Parameters
```bash
EPOCHS_RANGE=10,10        # min,max (same = fixed)
BATCH_OPTIONS=8,16,32,64  # Categorical choices
IMGSZ_OPTIONS=320,416,512,640
```

#### Hyperparameter Ranges
All ranges follow the pattern: `PARAM_RANGE=min,max`

**Learning Rate**:
- `LR0_RANGE`: Initial learning rate (log scale)
- `LRF_RANGE`: Final LR fraction
- `MOMENTUM_RANGE`: SGD momentum
- `WEIGHT_DECAY_RANGE`: L2 regularization

**Loss Weights**:
- `BOX_RANGE`: Bounding box loss weight
- `CLS_RANGE`: Classification loss weight
- `DFL_RANGE`: Distribution focal loss weight

**Augmentation - Color**:
- `HSV_H_RANGE`: Hue augmentation
- `HSV_S_RANGE`: Saturation
- `HSV_V_RANGE`: Value/brightness

**Augmentation - Geometric**:
- `DEGREES_RANGE`: Rotation angle
- `TRANSLATE_RANGE`: Translation fraction
- `SCALE_RANGE`: Scale variation
- `SHEAR_RANGE`: Shear angle
- `PERSPECTIVE_RANGE`: Perspective distortion

**Augmentation - Advanced**:
- `MOSAIC_RANGE`: Mosaic probability
- `MIXUP_RANGE`: Mixup probability
- `COPY_PASTE_RANGE`: Copy-paste probability
- `FLIPUD_RANGE`: Vertical flip probability
- `FLIPLR_RANGE`: Horizontal flip probability

## Data Flow

### Study Initialization

```
User runs: python run_study.py --preset quick --data data.yaml
                                       ↓
                        parse_args() reads arguments
                                       ↓
                        Load preset configuration
                                       ↓
                    Apply preset (CLI overrides)
                                       ↓
                    Set env vars for batch/imgsz
                                       ↓
                Create YOLOOptunaTrainer instance
                                       ↓
                    Load .env configuration
                                       ↓
                Parse hyperparameter ranges
                                       ↓
                Generate unique run_name
                                       ↓
                    Setup MLflow tracking
```

### Trial Execution

```
trainer.optimize() called
         ↓
Clear stale MLflow runs
         ↓
Create Optuna study
         ↓
┌────────────────────────────────┐
│  For each trial (n_trials):   │
│         ↓                      │
│  Optuna suggests params        │
│  (TPE sampler)                 │
│         ↓                      │
│  Start MLflow run (nested)     │
│         ↓                      │
│  Log parameters                │
│         ↓                      │
│  Load YOLO model               │
│         ↓                      │
│  Train model                   │
│  - Save to runs/optuna/        │
│         ↓                      │
│  Extract metrics               │
│         ↓                      │
│  Log metrics to MLflow         │
│         ↓                      │
│  Return optimization metric    │
│         ↓                      │
│  Optuna updates model          │
└────────────────────────────────┘
         ↓
Print best results
         ↓
Save best_hyperparameters.yaml
```

### Optuna's Learning Process

```
Trial 0: Random params
         ↓
     Run & measure
         ↓
Trial 1: Random params
         ↓
     Run & measure
         ↓
         ...
         ↓
Trial 5: Random params (exploration phase)
         ↓
     Run & measure
         ↓
Trial 6: TPE suggests params based on trials 0-5
         ↓
     Build probability model:
     - Which param values → good results?
     - Which param values → bad results?
         ↓
     Suggest params likely to improve
         ↓
     Run & measure
         ↓
Trial 7: TPE suggests params based on trials 0-6
         ↓
     Update probability model
         ↓
     Suggest better params (exploit good regions)
         ↓
         ...
         ↓
Converges to optimal hyperparameters
```

## Key Components

### Configuration Parser Functions

```python
def parse_range(env_var, default_range, as_int=False):
    """
    Parse min,max range from environment variable.

    Args:
        env_var: Environment variable name
        default_range: [min, max] if env var not set
        as_int: Convert to integers

    Returns:
        [min, max] as floats or ints
    """
```

```python
def parse_list(env_var, default_list):
    """
    Parse comma-separated list from environment variable.

    Args:
        env_var: Environment variable name
        default_list: Default list if env var not set

    Returns:
        List of strings
    """
```

### MLflow Integration

**Nested Runs Structure**:
```
Experiment: yolo-optuna-boost
├── Run: trial_0 (nested)
│   ├── Params: model=yolo11n, lr0=0.01, ...
│   └── Metrics: mAP50-95=0.65, ...
├── Run: trial_1 (nested)
│   ├── Params: model=yolo11s, lr0=0.005, ...
│   └── Metrics: mAP50-95=0.72, ...
└── ...
```

**Why nested?** All trials grouped under one experiment for easy comparison.

### Run Name Generation

```python
if run_name is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    self.run_name = f"run_{timestamp}"
else:
    self.run_name = run_name
```

**Result**: `run_20251221_143022` or custom name.

**Folder Structure**:
```
runs/optuna/{run_name}/trial_{N}/
                       ├── weights/
                       │   ├── best.pt
                       │   └── last.pt
                       ├── results.csv
                       └── args.yaml
```

## Configuration System

### Priority Cascade

When resolving a configuration value:

```python
# Example: device selection
device = (
    cli_arg          # --device 0 (highest priority)
    or preset_value  # preset['device'] if preset used
    or env_var       # DEVICE from .env
    or default       # 'cpu' (lowest priority)
)
```

### Environment Variable Loading

```python
# In train_yolo_optuna.py
from dotenv import load_dotenv
load_dotenv()  # Loads .env file into os.environ

# Then access with:
os.getenv('VARIABLE_NAME', 'default_value')
```

### Preset Application

```python
# In run_study.py main()
if args.preset:
    preset = get_preset(args.preset)

    # Only apply if not overridden by CLI
    if args.trials is None:
        args.trials = preset['n_trials']

    # Set env vars for trainer
    os.environ['BATCH_OPTIONS'] = preset['batch_options']
    os.environ['EPOCHS_RANGE'] = f"{preset['epochs']},{preset['epochs']}"
```

## Extending the Code

### Adding a New Preset

1. Edit `presets.py`:

```python
PRESETS['my_preset'] = {
    'description': 'My custom preset',
    'n_trials': 15,
    'epochs': 25,
    'models': ['yolo11n.pt', 'yolo11s.pt'],
    'optimization_metric': 'mAP50',
    'batch_options': '16,32',
    'imgsz_options': '416,512',
}
```

2. Use it:
```bash
python run_study.py --preset my_preset --data data.yaml
```

### Adding a New Optimization Metric

1. In `train_yolo_optuna.py`, add to the return logic in `objective()`:

```python
# After extracting metrics
if self.optimization_metric == 'my_metric':
    # Your custom calculation
    return custom_score
```

2. Document in `.env.example`:
```bash
# OPTIMIZATION_METRIC options:
# - my_metric: Description of what it optimizes
```

### Adding a New Hyperparameter

1. Add to `.env.example`:
```bash
MY_PARAM_RANGE=0.0,1.0
```

2. Add to `YOLOOptunaTrainer.__init__()`:
```python
self.ranges = {
    # ... existing ranges ...
    'my_param': parse_range('MY_PARAM_RANGE', [0.0, 1.0]),
}
```

3. Add to `objective()`:
```python
# Suggest value
my_param = trial.suggest_float('my_param', *self.ranges['my_param'])

# Log to MLflow
mlflow.log_param('my_param', my_param)

# Use in training
results = model.train(
    # ... other params ...
    my_param=my_param,
)
```

### Adding a New Command-Line Argument

1. In `run_study.py`, add to `parse_args()`:
```python
parser.add_argument('--my-arg', type=str, default=None,
                    help='Description of my argument')
```

2. Use in `main()`:
```python
trainer = YOLOOptunaTrainer(
    # ... other params ...
    my_param=args.my_arg
)
```

3. Add to `YOLOOptunaTrainer.__init__()`:
```python
self.my_param = my_param or os.getenv('MY_PARAM', 'default')
```

## Troubleshooting

### Common Issues

#### Issue: "Run with UUID XXX is already active"

**Cause**: Previous run didn't close properly (interrupted).

**Solution**: Already handled in code:
```python
# In optimize()
while mlflow.active_run():
    mlflow.end_run()
```

If still occurring, manually end runs:
```python
import mlflow
mlflow.end_run()
```

#### Issue: Trials overwriting each other

**Cause**: Old version didn't use unique run names.

**Solution**: Now using timestamped folders:
```python
project=f'runs/optuna/{self.run_name}'  # Unique per study
```

#### Issue: Out of memory during training

**Cause**: Batch size or image size too large.

**Solutions**:
1. Reduce batch size range in `.env`:
   ```bash
   BATCH_OPTIONS=4,8,16
   ```

2. Reduce image size range:
   ```bash
   IMGSZ_OPTIONS=320,416
   ```

3. Use smaller models:
   ```bash
   MODEL_VERSIONS=yolo11n.pt,yolo11s.pt
   ```

#### Issue: Optuna not learning (all trials similar)

**Cause**: Insufficient trials or narrow hyperparameter ranges.

**Solutions**:
1. Increase trials:
   ```bash
   python run_study.py --trials 50
   ```

2. Widen hyperparameter ranges in `.env`:
   ```bash
   LR0_RANGE=1e-6,1e-1  # Wider range
   ```

3. Check if using preset that limits models/params.

### Debugging Tips

#### Print trial parameters:

Add to `objective()` after parameter suggestion:
```python
print(f"\nTrial {trial.number} parameters:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")
```

#### Inspect MLflow data:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=['yolo-optuna-boost'])
print(runs[['params.model_version', 'metrics.mAP50_95']].head())
```

#### Resume interrupted study:

If using persistent storage:
```bash
# Study will automatically resume
python run_study.py --storage sqlite:///optuna.db --study-name my-study
```

#### Check Optuna study status:

```python
import optuna
study = optuna.load_study(
    study_name='yolo-optimization',
    storage='sqlite:///optuna.db'
)
print(f"Trials completed: {len(study.trials)}")
print(f"Best value: {study.best_value}")
```

## Performance Optimization

### Speed Up Trials

1. **Use GPU**:
   ```bash
   --device 0  # 10-20x faster than CPU
   ```

2. **Reduce epochs** for quick iteration:
   ```bash
   --preset quick  # or --epochs 5
   ```

3. **Smaller models** first:
   ```bash
   --models yolo11n.pt yolo11s.pt
   ```

4. **Parallel trials** (if you have multiple GPUs):
   ```python
   # Not currently implemented, but could use:
   # optuna.study.optimize(..., n_jobs=2)
   ```

### Reduce Memory Usage

1. **Smaller batch sizes**:
   ```bash
   BATCH_OPTIONS=4,8
   ```

2. **Smaller image sizes**:
   ```bash
   IMGSZ_OPTIONS=320,416
   ```

3. **Gradient accumulation** (modify YOLO train call):
   ```python
   # In objective():
   results = model.train(
       # ... other params ...
       accumulate=2,  # Simulate larger batch
   )
   ```

## Testing

### Unit Tests (Recommended to Add)

Example test structure:

```python
# tests/test_presets.py
def test_get_preset():
    preset = get_preset('quick')
    assert preset['n_trials'] == 5
    assert 'yolo11n.pt' in preset['models']

# tests/test_trainer.py
def test_parse_range():
    result = parse_range('TEST_VAR', [0, 1])
    assert len(result) == 2

def test_trainer_init():
    trainer = YOLOOptunaTrainer(data_yaml='data.yaml')
    assert trainer.device == 'cpu'  # default
```

### Integration Testing

Test full workflow:

```bash
# Quick test with minimal trials
python run_study.py --preset quick --trials 2 --epochs 1 --data data.yaml
```

## Future Enhancements

Potential improvements:

1. **Multi-objective optimization**:
   - Optimize accuracy AND speed simultaneously
   - Pareto frontier visualization

2. **Early stopping**:
   - Stop trials that perform poorly early
   - Save compute time

3. **Transfer learning**:
   - Warm-start Optuna from previous studies
   - Share knowledge across datasets

4. **Automatic data analysis**:
   - Generate plots of hyperparameter importance
   - Correlation analysis

5. **Hyperband/BOHB**:
   - More efficient sampling strategies
   - Adaptive resource allocation

6. **Distributed optimization**:
   - Run trials across multiple machines
   - Kubernetes integration

---

## Contributing

When modifying the codebase:

1. **Document changes** in this file
2. **Update README.md** if user-facing
3. **Test with quick preset** before committing
4. **Use meaningful commit messages**

## License

MIT License - See LICENSE file
