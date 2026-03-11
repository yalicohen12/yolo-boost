import optuna
import mlflow
from ultralytics import YOLO
import yaml
from pathlib import Path
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable Ultralytics' built-in MLflow integration — it conflicts with our setup
# by registering its own callbacks on the Trainer and trying to write to /mlflow
try:
    from ultralytics.utils import SETTINGS
    SETTINGS['mlflow'] = False
except Exception:
    pass


# YOLO11/12 default values — used when a param is outside the focused search space
# YOLO12 uses the same Ultralytics training API and the same hyperparameter defaults
YOLO11_DEFAULTS = {
    'optimizer':        'SGD',
    'lr0':              0.01,
    'lrf':              0.01,
    'momentum':         0.937,
    'weight_decay':     0.0005,
    'warmup_epochs':    3,
    'warmup_momentum':  0.8,
    'warmup_bias_lr':   0.1,
    'box':              7.5,
    'cls':              0.5,
    'dfl':              1.5,
    'label_smoothing':  0.0,
    'hsv_h':            0.015,
    'hsv_s':            0.7,
    'hsv_v':            0.4,
    'degrees':          0.0,
    'translate':        0.1,
    'scale':            0.5,
    'shear':            0.0,
    'perspective':      0.0,
    'flipud':           0.0,
    'fliplr':           0.5,
    'mosaic':           1.0,
    'mixup':            0.0,
    'copy_paste':       0.0,
    'erasing':          0.4,
    'bgr':              0.0,
    'close_mosaic':     10,
}


def parse_range(env_var, default_range, as_int=False):
    """Parse range from environment variable."""
    value = os.getenv(env_var, ','.join(map(str, default_range)))
    parsed = [float(x) for x in value.split(',')]
    return [int(x) for x in parsed] if as_int else parsed


def parse_list(env_var, default_list):
    """Parse comma-separated list from environment variable."""
    value = os.getenv(env_var, ','.join(default_list))
    return [x.strip() for x in value.split(',')]


def auto_detect_image_size(data_yaml_path):
    """Auto-detect image size from dataset.

    Reads the data.yaml file and finds an actual image to determine dimensions.
    Returns the max dimension (YOLO uses square images).
    """
    from PIL import Image
    import glob

    try:
        # Load data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Get dataset root and train path
        dataset_root = Path(data_config.get('path', '.'))
        train_path = data_config.get('train', 'images/train')

        # Construct full path to training images
        if not dataset_root.is_absolute():
            # Relative to data.yaml location
            dataset_root = Path(data_yaml_path).parent / dataset_root

        train_images_path = dataset_root / train_path

        # Find first image
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_file = None

        for ext in image_extensions:
            images = list(train_images_path.glob(ext))
            if images:
                image_file = images[0]
                break

        if image_file is None:
            print(f"Warning: No images found in {train_images_path}, using default size 640")
            return 640

        # Read image dimensions
        with Image.open(image_file) as img:
            width, height = img.size
            # YOLO uses square images, take max dimension and round to nearest 32
            max_dim = max(width, height)
            # Round to nearest 32 (YOLO requirement)
            imgsz = ((max_dim + 31) // 32) * 32

            print(f"Auto-detected image size from {image_file.name}: {width}x{height} -> using {imgsz}")
            return imgsz

    except Exception as e:
        print(f"Warning: Could not auto-detect image size ({e}), using default 640")
        return 640


class YOLOOptunaTrainer:
    def __init__(self, data_yaml=None, mlflow_tracking_uri=None, experiment_name=None,
                 model_versions=None, device=None, optimization_metric=None, run_name=None):
        # Load configuration from environment variables with fallback to parameters
        self.data_yaml = data_yaml or os.getenv('DATA_YAML', 'data.yaml')
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv('MLFLOW_TRACKING_URI', './mlruns')
        self.experiment_name = experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME', 'yolo-optuna-boost')
        self.device = device or os.getenv('DEVICE', 'cpu')
        self.optimization_metric = optimization_metric or os.getenv('OPTIMIZATION_METRIC', 'mAP50-95')

        # Generate unique run name if not provided (timestamp-based)
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_name = f"run_{timestamp}"
        else:
            self.run_name = run_name

        # Model versions will be optimized by Optuna
        self.model_versions = model_versions or parse_list('MODEL_VERSIONS',
            ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt',
             'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt'])

        # Fixed training settings (not searched by Optuna — set per preset)
        self.epochs = int(os.getenv('EPOCHS', '50'))
        self.patience = int(os.getenv('PATIENCE', '50'))
        self.optimizer_options = parse_list('OPTIMIZER_OPTIONS', ['SGD', 'Adam', 'AdamW', 'NAdam'])

        # Focused search: set of param names to search. None = search all params.
        # When a param is not in the set, YOLO11_DEFAULTS is used instead.
        raw = os.getenv('SEARCH_PARAMS', '')
        self.search_params = set(p.strip() for p in raw.split(',') if p.strip()) or None

        # Hyperparameter search ranges
        self.ranges = {
            # Learning rate
            'lr0': parse_range('LR0_RANGE', [1e-5, 1e-1]),
            'lrf': parse_range('LRF_RANGE', [0.01, 1.0]),
            'momentum': parse_range('MOMENTUM_RANGE', [0.8, 0.99]),        # fixed: was [0.6, 0.98]
            'weight_decay': parse_range('WEIGHT_DECAY_RANGE', [0.0, 0.01]),  # fixed: was [0.0, 0.001]
            # Warmup
            'warmup_epochs': parse_range('WARMUP_EPOCHS_RANGE', [0, 5], as_int=True),
            'warmup_momentum': parse_range('WARMUP_MOMENTUM_RANGE', [0.0, 0.95]),
            'warmup_bias_lr': parse_range('WARMUP_BIAS_LR_RANGE', [0.0, 0.2]),
            # Loss weights — calibrated for YOLO11 defaults (box=7.5, cls=0.5, dfl=1.5)
            'box': parse_range('BOX_RANGE', [1.0, 20.0]),   # fixed: was [0.02, 0.2] — completely wrong for YOLO11
            'cls': parse_range('CLS_RANGE', [0.1, 4.0]),    # fixed: was [0.2, 4.0]
            'dfl': parse_range('DFL_RANGE', [0.5, 4.0]),    # fixed: was [0.4, 2.0]
            # Regularization
            'label_smoothing': parse_range('LABEL_SMOOTHING_RANGE', [0.0, 0.1]),
            # Augmentation — color
            'hsv_h': parse_range('HSV_H_RANGE', [0.0, 0.1]),
            'hsv_s': parse_range('HSV_S_RANGE', [0.0, 0.9]),
            'hsv_v': parse_range('HSV_V_RANGE', [0.0, 0.9]),
            # Augmentation — geometric
            'degrees': parse_range('DEGREES_RANGE', [0.0, 45.0]),
            'translate': parse_range('TRANSLATE_RANGE', [0.0, 0.5]),   # fixed: was [0.0, 0.9] — too wide
            'scale': parse_range('SCALE_RANGE', [0.0, 0.9]),
            'shear': parse_range('SHEAR_RANGE', [0.0, 10.0]),
            'perspective': parse_range('PERSPECTIVE_RANGE', [0.0, 0.001]),
            'flipud': parse_range('FLIPUD_RANGE', [0.0, 1.0]),
            'fliplr': parse_range('FLIPLR_RANGE', [0.0, 1.0]),
            # Augmentation — advanced
            'mosaic': parse_range('MOSAIC_RANGE', [0.0, 1.0]),
            'mixup': parse_range('MIXUP_RANGE', [0.0, 1.0]),
            'copy_paste': parse_range('COPY_PASTE_RANGE', [0.0, 1.0]),
            'erasing': parse_range('ERASING_RANGE', [0.0, 0.9]),
            'bgr': parse_range('BGR_RANGE', [0.0, 1.0]),
            # Mosaic scheduling
            'close_mosaic': parse_range('CLOSE_MOSAIC_RANGE', [0, 20], as_int=True),
        }

        # Auto-detect image size from dataset (fixed, not optimized)
        self.imgsz = auto_detect_image_size(self.data_yaml)
        self.batch_options = [int(x) for x in parse_list('BATCH_OPTIONS', ['8', '16', '32', '64'])]

        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.enable_system_metrics_logging()  # CPU, RAM, GPU utilization per run

    def objective(self, trial):
        """
        Optuna objective function for YOLO hyperparameter optimization.
        All ranges are loaded from .env file.
        """
        # Helper: suggest if param is in the active search space, else use YOLO11 default.
        # Enables focused search mode (SEARCH_PARAMS env / focused preset).
        def _p(name, suggest_fn, *args, **kwargs):
            if self.search_params is None or name in self.search_params:
                return suggest_fn(name, *args, **kwargs)
            if name == 'model_version':
                return self.model_versions[0]
            if name == 'batch':
                return self.batch_options[0]
            return YOLO11_DEFAULTS[name]

        # Model, optimizer, batch (categorical)
        model_name = _p('model_version', trial.suggest_categorical, self.model_versions)
        optimizer   = _p('optimizer',     trial.suggest_categorical, self.optimizer_options)
        batch       = _p('batch',         trial.suggest_categorical, self.batch_options)
        imgsz = self.imgsz  # Auto-detected and fixed

        # Learning rate & optimizer settings
        lr0          = _p('lr0',          trial.suggest_float, *self.ranges['lr0'], log=True)
        lrf          = _p('lrf',          trial.suggest_float, *self.ranges['lrf'])
        momentum     = _p('momentum',     trial.suggest_float, *self.ranges['momentum'])
        weight_decay = _p('weight_decay', trial.suggest_float, *self.ranges['weight_decay'])

        # Warmup
        warmup_epochs   = _p('warmup_epochs',   trial.suggest_int,   *self.ranges['warmup_epochs'])
        warmup_momentum = _p('warmup_momentum', trial.suggest_float, *self.ranges['warmup_momentum'])
        warmup_bias_lr  = _p('warmup_bias_lr',  trial.suggest_float, *self.ranges['warmup_bias_lr'])

        # Loss weights
        box = _p('box', trial.suggest_float, *self.ranges['box'])
        cls = _p('cls', trial.suggest_float, *self.ranges['cls'])
        dfl = _p('dfl', trial.suggest_float, *self.ranges['dfl'])

        # Regularization
        label_smoothing = _p('label_smoothing', trial.suggest_float, *self.ranges['label_smoothing'])

        # Augmentation — color
        hsv_h = _p('hsv_h', trial.suggest_float, *self.ranges['hsv_h'])
        hsv_s = _p('hsv_s', trial.suggest_float, *self.ranges['hsv_s'])
        hsv_v = _p('hsv_v', trial.suggest_float, *self.ranges['hsv_v'])

        # Augmentation — geometric
        degrees     = _p('degrees',     trial.suggest_float, *self.ranges['degrees'])
        translate   = _p('translate',   trial.suggest_float, *self.ranges['translate'])
        scale       = _p('scale',       trial.suggest_float, *self.ranges['scale'])
        shear       = _p('shear',       trial.suggest_float, *self.ranges['shear'])
        perspective = _p('perspective', trial.suggest_float, *self.ranges['perspective'])
        flipud      = _p('flipud',      trial.suggest_float, *self.ranges['flipud'])
        fliplr      = _p('fliplr',      trial.suggest_float, *self.ranges['fliplr'])

        # Augmentation — advanced
        mosaic       = _p('mosaic',       trial.suggest_float, *self.ranges['mosaic'])
        mixup        = _p('mixup',        trial.suggest_float, *self.ranges['mixup'])
        copy_paste   = _p('copy_paste',   trial.suggest_float, *self.ranges['copy_paste'])
        erasing      = _p('erasing',      trial.suggest_float, *self.ranges['erasing'])
        bgr          = _p('bgr',          trial.suggest_float, *self.ranges['bgr'])
        close_mosaic = _p('close_mosaic', trial.suggest_int,   *self.ranges['close_mosaic'])

        # Start MLflow run with meaningful name
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log trial number
            mlflow.log_param('trial_number', trial.number)

            # Log all hyperparameters
            mlflow.log_params({
                'model_version': model_name,
                'optimizer': optimizer,
                'epochs': self.epochs,
                'patience': self.patience,
                'imgsz': imgsz,
                'batch': batch,
                'lr0': lr0,
                'lrf': lrf,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'warmup_epochs': warmup_epochs,
                'warmup_momentum': warmup_momentum,
                'warmup_bias_lr': warmup_bias_lr,
                'box': box,
                'cls': cls,
                'dfl': dfl,
                'label_smoothing': label_smoothing,
                'hsv_h': hsv_h,
                'hsv_s': hsv_s,
                'hsv_v': hsv_v,
                'degrees': degrees,
                'translate': translate,
                'scale': scale,
                'shear': shear,
                'perspective': perspective,
                'flipud': flipud,
                'fliplr': fliplr,
                'mosaic': mosaic,
                'mixup': mixup,
                'copy_paste': copy_paste,
                'erasing': erasing,
                'bgr': bgr,
                'close_mosaic': close_mosaic,
            })

            # Pruning callback — reports val mAP50-95 to Optuna each epoch
            # If Optuna decides to prune, sets trainer.stop=True to end training early
            pruned = False

            def on_fit_epoch_end(trainer):
                nonlocal pruned
                epoch = trainer.epoch
                val_map95 = trainer.metrics.get('metrics/mAP50-95(B)', 0.0)

                # Log per-epoch metrics to MLflow — produces training curves per trial
                epoch_metrics = {
                    'epoch_mAP50_95':  val_map95,
                    'epoch_mAP50':     trainer.metrics.get('metrics/mAP50(B)', 0.0),
                    'epoch_precision': trainer.metrics.get('metrics/precision(B)', 0.0),
                    'epoch_recall':    trainer.metrics.get('metrics/recall(B)', 0.0),
                }
                # Add training losses if available
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    losses = trainer.loss_items.detach().cpu().tolist() \
                             if hasattr(trainer.loss_items, 'detach') \
                             else list(trainer.loss_items)
                    loss_names = ['train_box_loss', 'train_cls_loss', 'train_dfl_loss']
                    for name, val in zip(loss_names, losses):
                        epoch_metrics[name] = val
                mlflow.log_metrics(epoch_metrics, step=epoch)

                # Report to Optuna for pruning decisions
                trial.report(val_map95, step=epoch)
                if trial.should_prune():
                    pruned = True
                    trainer.stop = True

            model = YOLO(model_name)
            model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

            # Train
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                patience=self.patience,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                optimizer=optimizer,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                box=box,
                cls=cls,
                dfl=dfl,
                label_smoothing=label_smoothing,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                erasing=erasing,
                bgr=bgr,
                close_mosaic=close_mosaic,
                project=f'runs/optuna/{self.run_name}',
                name=f'trial_{trial.number}',
                exist_ok=False,
                verbose=True,
            )

            # Raise TrialPruned if callback signalled early stop
            if pruned:
                raise optuna.TrialPruned()

            # Get validation metrics
            metrics = results.results_dict
            map50 = metrics.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
            precision = metrics.get('metrics/precision(B)', 0)
            recall = metrics.get('metrics/recall(B)', 0)

            # Real measured inference speed from YOLO (ms per image)
            # Normalize: 0ms=1.0 (perfect), 100ms=0.0 (slow baseline)
            inference_ms = results.speed.get('inference', 999)
            speed_score = max(0.0, 1.0 - inference_ms / 100.0)

            # F1 score
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            # Log metrics to MLflow
            mlflow.log_metrics({
                'mAP50': map50,
                'mAP50_95': map50_95,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'inference_ms': inference_ms,
                'speed_score': speed_score,
            })

            # Store in trial for parent run progression logging
            trial.set_user_attr('mAP50', map50)
            trial.set_user_attr('mAP50_95', map50_95)
            trial.set_user_attr('precision', precision)
            trial.set_user_attr('recall', recall)
            trial.set_user_attr('f1', f1)
            trial.set_user_attr('inference_ms', inference_ms)
            trial.set_user_attr('speed_score', speed_score)

            # Log model weights
            trial_dir = Path(f'runs/optuna/{self.run_name}/trial_{trial.number}')
            model_path = trial_dir / 'weights' / 'best.pt'
            if model_path.exists():
                self._safe_log_artifact(model_path, artifact_path='weights')

            # Log all YOLO-generated visualizations
            for artifact_name in [
                'results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                'PR_curve.png', 'F1_curve.png', 'R_curve.png', 'P_curve.png',
                'labels.jpg', 'val_batch0_pred.jpg', 'val_batch1_pred.jpg',
            ]:
                artifact_path = trial_dir / artifact_name
                if artifact_path.exists():
                    self._safe_log_artifact(artifact_path, artifact_path='plots')

        # Return optimization metric based on configuration
        if self.optimization_metric == 'mAP50':
            return map50
        elif self.optimization_metric == 'mAP50-95':
            return map50_95
        elif self.optimization_metric == 'precision':
            return precision
        elif self.optimization_metric == 'recall':
            return recall
        elif self.optimization_metric == 'speed':
            return map50_95 * 0.5 + speed_score * 0.5
        elif self.optimization_metric == 'balanced':
            return map50_95 * 0.7 + speed_score * 0.3
        else:
            return map50_95  # Default

    def optimize(self, n_trials=20, study_name='yolo-optimization', storage=None):
        """
        Run Optuna optimization study.

        Args:
            n_trials: Number of optimization trials
            study_name: Name of the Optuna study
            storage: Database URL for Optuna storage (optional)
        """
        # End any stale MLflow runs from previous interruptions
        while mlflow.active_run():
            mlflow.end_run()

        # Create Optuna study with median pruner
        # n_startup_trials: don't prune until 5 trials complete (need baseline data)
        # n_warmup_steps: don't prune a trial's first 10 epochs (models need warmup)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
        )

        # Create parent MLflow run for the entire study
        with mlflow.start_run(run_name=f"{study_name}_{self.run_name}") as parent_run:
            mlflow.log_param('study_name', study_name)
            mlflow.log_param('n_trials', n_trials)
            mlflow.log_param('run_name', self.run_name)
            mlflow.log_param('optimization_metric', self.optimization_metric)
            mlflow.log_param('search_params', ','.join(sorted(self.search_params)) if self.search_params else 'all')

            # Log dataset config for full reproducibility
            if Path(self.data_yaml).exists():
                self._safe_log_artifact(self.data_yaml, artifact_path='dataset')

            # Store parent run ID for logging trial metrics
            self._parent_run_id = parent_run.info.run_id

            # Run optimization (each trial creates nested run)
            study.optimize(
                self.objective,
                n_trials=n_trials,
                callbacks=[self._trial_callback],
            )

            # Log best trial's metrics as plain scalars — these show as columns
            # in the MLflow experiment table, making runs comparable at a glance
            best_t = study.best_trial
            n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            mlflow.log_metrics({
                'best_mAP50':        best_t.user_attrs.get('mAP50', 0),
                'best_mAP50_95':     best_t.user_attrs.get('mAP50_95', 0),
                'best_precision':    best_t.user_attrs.get('precision', 0),
                'best_recall':       best_t.user_attrs.get('recall', 0),
                'best_f1':           best_t.user_attrs.get('f1', 0),
                'best_inference_ms': best_t.user_attrs.get('inference_ms', 999),
                'best_speed_score':  best_t.user_attrs.get('speed_score', 0),
                'n_trials_complete': n_complete,
                'n_trials_pruned':   n_pruned,
            })
            mlflow.log_param('best_trial_number', best_t.number)
            mlflow.log_params({f'best_{k}': v for k, v in study.best_params.items()})

            # Log per-trial final metrics as a time-series on the parent run
            # (step = trial number) — shows study progression: did each trial do better?
            # Prefixed with "trial_" to distinguish from epoch-level metrics in child runs
            running_best = 0.0
            for t in sorted(study.trials, key=lambda x: x.number):
                if t.state == optuna.trial.TrialState.COMPLETE:
                    step = t.number
                    trial_map95 = t.user_attrs.get('mAP50_95', 0)
                    running_best = max(running_best, trial_map95)
                    mlflow.log_metrics({
                        'trial_mAP50':        t.user_attrs.get('mAP50', 0),
                        'trial_mAP50_95':     trial_map95,
                        'trial_precision':    t.user_attrs.get('precision', 0),
                        'trial_recall':       t.user_attrs.get('recall', 0),
                        'trial_f1':           t.user_attrs.get('f1', 0),
                        'trial_inference_ms': t.user_attrs.get('inference_ms', 0),
                        'trial_speed_score':  t.user_attrs.get('speed_score', 0),
                        'best_so_far':        running_best,
                    }, step=step)

            # Must be inside the with block — calling log_artifact outside an
            # active run causes MLflow to auto-create a blank run with a random name
            self._log_optuna_plots(study)

        # Print best results
        print('\n' + '='*50)
        print('Optimization completed!')
        print('='*50)
        print(f'Trials completed: {n_complete}  |  Pruned (saved compute): {n_pruned}')
        print(f'Best trial: {study.best_trial.number}')
        print(f'Best {self.optimization_metric}: {study.best_value:.4f}')
        print('\nBest hyperparameters:')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')

        # Save best hyperparameters
        best_params_path = Path('best_hyperparameters.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        print(f'\nBest hyperparameters saved to {best_params_path}')

        return study

    def _safe_log_artifact(self, local_path, artifact_path=None):
        """Log artifact, with a clear message if the server artifact store isn't reachable.

        When using a remote MLflow server without --serve-artifacts, log_artifact()
        tries to write to the server's local artifact path (e.g. /mlflow) from the
        client machine, which fails with PermissionError. Fix: run the server with
        --serve-artifacts, or use a shared artifact store (S3, GCS, etc.).
        """
        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except PermissionError:
            if not getattr(self, '_artifact_warning_shown', False):
                print('\nNote: Binary artifact upload skipped (weights, plots). '
                      'To enable, restart MLflow with: mlflow server --serve-artifacts\n')
                self._artifact_warning_shown = True

    def _trial_callback(self, study, trial):
        """Print a concise summary after each trial completes or is pruned."""
        n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            is_best = trial.value == study.best_value
            marker = '  ← NEW BEST' if is_best else f'  (best so far: {study.best_value:.4f})'
            print(f'\n{"─"*60}')
            print(f'  Trial {trial.number} done  |  score: {trial.value:.4f}{marker}')
            print(f'  mAP50-95={trial.user_attrs.get("mAP50_95", 0):.4f}  '
                  f'mAP50={trial.user_attrs.get("mAP50", 0):.4f}  '
                  f'F1={trial.user_attrs.get("f1", 0):.4f}  '
                  f'inference={trial.user_attrs.get("inference_ms", 0):.1f}ms')
            print(f'  model={trial.params.get("model_version", "?")}  '
                  f'optimizer={trial.params.get("optimizer", "?")}  '
                  f'lr0={trial.params.get("lr0", 0):.5f}  '
                  f'batch={trial.params.get("batch", "?")}')
            print(f'  completed: {n_complete} trials')
            print(f'{"─"*60}')
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f'\n  Trial {trial.number} pruned  |  best so far: {study.best_value:.4f}')

    def _log_optuna_plots(self, study):
        """Generate Optuna visualization plots and log to MLflow as HTML artifacts."""
        import tempfile
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice,
            )
            plots = {
                'optimization_history': plot_optimization_history(study),
                'param_importances':    plot_param_importances(study),
                'parallel_coordinate':  plot_parallel_coordinate(study),
                'slice':                plot_slice(study),
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                for name, fig in plots.items():
                    html_path = os.path.join(tmpdir, f'{name}.html')
                    fig.write_html(html_path)
                    self._safe_log_artifact(html_path, artifact_path='optuna_plots')
            print('Optuna plots saved to MLflow under optuna_plots/')
        except Exception as e:
            print(f'Warning: Could not generate Optuna plots ({e}). '
                  f'Install plotly: pip install plotly')

    def train_baseline(self, model_name=None, epochs=None):
        """
        Run a single baseline training without optimization.
        Useful for comparing with optimized results.

        Args:
            model_name: YOLO model to use (default: first in model_versions list)
            epochs: Number of epochs (default: from EPOCHS_RANGE)
        """
        # Use defaults if not specified
        if model_name is None:
            model_name = self.model_versions[0]
        if epochs is None:
            epochs = self.epochs

        print(f"\nRunning baseline training: {model_name}, {epochs} epochs\n")

        # Start parent MLflow run
        with mlflow.start_run(run_name=f"baseline_{self.run_name}"):
            mlflow.log_param('run_type', 'baseline')
            mlflow.log_param('model_version', model_name)
            mlflow.log_param('epochs', epochs)
            mlflow.log_param('device', self.device)
            mlflow.log_param('run_name', self.run_name)

            # Load model
            model = YOLO(model_name)

            # Train with YOLO defaults (no custom hyperparameters)
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                device=self.device,
                project=f'runs/optuna/{self.run_name}',
                name='baseline',
                exist_ok=False,
                verbose=True,
            )

            # Get validation metrics
            metrics = results.results_dict
            map50 = metrics.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
            precision = metrics.get('metrics/precision(B)', 0)
            recall = metrics.get('metrics/recall(B)', 0)

            # Log metrics to MLflow
            mlflow.log_metrics({
                'mAP50': map50,
                'mAP50_95': map50_95,
                'precision': precision,
                'recall': recall,
            })

            # Log model artifacts
            model_path = Path(f'runs/optuna/{self.run_name}/baseline/weights/best.pt')
            if model_path.exists():
                self._safe_log_artifact(model_path, artifact_path='weights')

            print(f"\nBaseline Results:")
            print(f"  mAP50: {map50:.4f}")
            print(f"  mAP50-95: {map50_95:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")


def main():
    """
    Example usage of YOLOOptunaTrainer.
    Model version/size is now automatically optimized by Optuna.
    """
    trainer = YOLOOptunaTrainer(
        data_yaml='data.yaml',
        mlflow_tracking_uri='http://localhost:5000',
        experiment_name='yolo-optuna-boost',
        model_versions=['yolo11n.pt', 'yolo11s.pt', 'yolo12n.pt', 'yolo12s.pt']
    )

    study = trainer.optimize(
        n_trials=20,
        study_name='yolo-optimization',
        storage=None  # Use 'sqlite:///optuna_study.db' for persistent storage
    )


if __name__ == '__main__':
    main()
