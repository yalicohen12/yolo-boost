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


def parse_range(env_var, default_range, as_int=False):
    """Parse range from environment variable."""
    value = os.getenv(env_var, ','.join(map(str, default_range)))
    parsed = [float(x) for x in value.split(',')]
    return [int(x) for x in parsed] if as_int else parsed


def parse_list(env_var, default_list):
    """Parse comma-separated list from environment variable."""
    value = os.getenv(env_var, ','.join(default_list))
    return [x.strip() for x in value.split(',')]


class YOLOOptunaTrainer:
    def __init__(self, data_yaml=None, mlflow_tracking_uri=None, experiment_name=None,
                 model_versions=None, device=None, optimization_metric=None, run_name=None):
        # Load configuration from environment variables with fallback to parameters
        self.data_yaml = data_yaml or os.getenv('DATA_YAML', 'data.yaml')
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
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
            ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt'])

        # Load hyperparameter ranges from environment
        self.ranges = {
            'lr0': parse_range('LR0_RANGE', [1e-5, 1e-1]),
            'lrf': parse_range('LRF_RANGE', [0.01, 1.0]),
            'momentum': parse_range('MOMENTUM_RANGE', [0.6, 0.98]),
            'weight_decay': parse_range('WEIGHT_DECAY_RANGE', [0.0, 0.001]),
            'warmup_epochs': parse_range('WARMUP_EPOCHS_RANGE', [0, 5], as_int=True),
            'warmup_momentum': parse_range('WARMUP_MOMENTUM_RANGE', [0.0, 0.95]),
            'box': parse_range('BOX_RANGE', [0.02, 0.2]),
            'cls': parse_range('CLS_RANGE', [0.2, 4.0]),
            'dfl': parse_range('DFL_RANGE', [0.4, 2.0]),
            'hsv_h': parse_range('HSV_H_RANGE', [0.0, 0.1]),
            'hsv_s': parse_range('HSV_S_RANGE', [0.0, 0.9]),
            'hsv_v': parse_range('HSV_V_RANGE', [0.0, 0.9]),
            'degrees': parse_range('DEGREES_RANGE', [0.0, 45.0]),
            'translate': parse_range('TRANSLATE_RANGE', [0.0, 0.9]),
            'scale': parse_range('SCALE_RANGE', [0.0, 0.9]),
            'shear': parse_range('SHEAR_RANGE', [0.0, 10.0]),
            'perspective': parse_range('PERSPECTIVE_RANGE', [0.0, 0.001]),
            'flipud': parse_range('FLIPUD_RANGE', [0.0, 1.0]),
            'fliplr': parse_range('FLIPLR_RANGE', [0.0, 1.0]),
            'mosaic': parse_range('MOSAIC_RANGE', [0.0, 1.0]),
            'mixup': parse_range('MIXUP_RANGE', [0.0, 1.0]),
            'copy_paste': parse_range('COPY_PASTE_RANGE', [0.0, 1.0]),
            'epochs': parse_range('EPOCHS_RANGE', [50, 200], as_int=True),
        }

        self.imgsz_options = [int(x) for x in parse_list('IMGSZ_OPTIONS', ['320', '416', '512', '640'])]
        self.batch_options = [int(x) for x in parse_list('BATCH_OPTIONS', ['8', '16', '32', '64'])]

        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def objective(self, trial):
        """
        Optuna objective function for YOLO hyperparameter optimization.
        All ranges are loaded from .env file.
        """
        # Suggest model version/size
        model_name = trial.suggest_categorical('model_version', self.model_versions)

        # Suggest hyperparameters (ranges from .env or defaults from official Ultralytics docs)
        lr0 = trial.suggest_float('lr0', *self.ranges['lr0'], log=True)
        lrf = trial.suggest_float('lrf', *self.ranges['lrf'])
        momentum = trial.suggest_float('momentum', *self.ranges['momentum'])
        weight_decay = trial.suggest_float('weight_decay', *self.ranges['weight_decay'])
        warmup_epochs = trial.suggest_int('warmup_epochs', *self.ranges['warmup_epochs'])
        warmup_momentum = trial.suggest_float('warmup_momentum', *self.ranges['warmup_momentum'])
        box = trial.suggest_float('box', *self.ranges['box'])
        cls = trial.suggest_float('cls', *self.ranges['cls'])
        dfl = trial.suggest_float('dfl', *self.ranges['dfl'])

        # Image size and batch size
        imgsz = trial.suggest_categorical('imgsz', self.imgsz_options)
        batch = trial.suggest_categorical('batch', self.batch_options)

        # Data augmentation
        hsv_h = trial.suggest_float('hsv_h', *self.ranges['hsv_h'])
        hsv_s = trial.suggest_float('hsv_s', *self.ranges['hsv_s'])
        hsv_v = trial.suggest_float('hsv_v', *self.ranges['hsv_v'])
        degrees = trial.suggest_float('degrees', *self.ranges['degrees'])
        translate = trial.suggest_float('translate', *self.ranges['translate'])
        scale = trial.suggest_float('scale', *self.ranges['scale'])
        shear = trial.suggest_float('shear', *self.ranges['shear'])
        perspective = trial.suggest_float('perspective', *self.ranges['perspective'])
        flipud = trial.suggest_float('flipud', *self.ranges['flipud'])
        fliplr = trial.suggest_float('fliplr', *self.ranges['fliplr'])
        mosaic = trial.suggest_float('mosaic', *self.ranges['mosaic'])
        mixup = trial.suggest_float('mixup', *self.ranges['mixup'])
        copy_paste = trial.suggest_float('copy_paste', *self.ranges['copy_paste'])

        # Start MLflow run with meaningful name
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log trial number
            mlflow.log_param('trial_number', trial.number)

            # Log all hyperparameters
            mlflow.log_params({
                'model_version': model_name,
                'lr0': lr0,
                'lrf': lrf,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'warmup_epochs': warmup_epochs,
                'warmup_momentum': warmup_momentum,
                'box': box,
                'cls': cls,
                'dfl': dfl,
                'imgsz': imgsz,
                'batch': batch,
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
            })

            # Load model (using Optuna-suggested model version)
            model = YOLO(model_name)

            # Train
            epochs = trial.suggest_int('epochs', *self.ranges['epochs'])
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                box=box,
                cls=cls,
                dfl=dfl,
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
                project=f'runs/optuna/{self.run_name}',
                name=f'trial_{trial.number}',
                exist_ok=False,  # Don't overwrite - fail if exists (shouldn't happen with unique folders)
                verbose=True,
            )

            # Get validation metrics
            metrics = results.results_dict
            map50 = metrics.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
            precision = metrics.get('metrics/precision(B)', 0)
            recall = metrics.get('metrics/recall(B)', 0)

            # Calculate inference speed (smaller models = faster)
            model_size_score = {'yolo11n.pt': 1.0, 'yolo11s.pt': 0.8, 'yolo11m.pt': 0.6,
                              'yolo11l.pt': 0.4, 'yolo11x.pt': 0.2,
                              'yolov8n.pt': 1.0, 'yolov8s.pt': 0.8, 'yolov8m.pt': 0.6,
                              'yolov8l.pt': 0.4, 'yolov8x.pt': 0.2}.get(model_name, 0.5)

            # Log metrics to MLflow
            mlflow.log_metrics({
                'mAP50': map50,
                'mAP50_95': map50_95,
                'precision': precision,
                'recall': recall,
                'speed_score': model_size_score,
            })

            # Log model artifacts
            model_path = Path(f'runs/optuna/{self.run_name}/trial_{trial.number}/weights/best.pt')
            if model_path.exists():
                mlflow.log_artifact(str(model_path))

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
            # Optimize for speed (balance between accuracy and model size)
            return map50_95 * 0.5 + model_size_score * 0.5
        elif self.optimization_metric == 'balanced':
            # Balanced: 70% accuracy, 30% speed
            return map50_95 * 0.7 + model_size_score * 0.3
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

        # Create Optuna study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            storage=storage,
            load_if_exists=True,
        )

        # Create parent MLflow run for the entire study
        with mlflow.start_run(run_name=f"{study_name}_{self.run_name}"):
            mlflow.log_param('study_name', study_name)
            mlflow.log_param('n_trials', n_trials)
            mlflow.log_param('run_name', self.run_name)
            mlflow.log_param('optimization_metric', self.optimization_metric)

            # Run optimization (each trial creates nested run)
            study.optimize(
                self.objective,
                n_trials=n_trials,
            )

            # Log best results to parent run
            mlflow.log_metric('best_value', study.best_value)
            mlflow.log_params({f'best_{k}': v for k, v in study.best_params.items()})

        # Print best results
        print('\n' + '='*50)
        print('Optimization completed!')
        print('='*50)
        print(f'Best trial: {study.best_trial.number}')
        print(f'Best mAP50-95: {study.best_value:.4f}')
        print('\nBest hyperparameters:')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')

        # Save best hyperparameters
        best_params_path = Path('best_hyperparameters.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(study.best_params, f, default_flow_style=False)
        print(f'\nBest hyperparameters saved to {best_params_path}')

        return study

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
            epochs = self.ranges['epochs'][0]  # Use min of range

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
                mlflow.log_artifact(str(model_path))

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
        model_versions=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']  # Models to try
    )

    study = trainer.optimize(
        n_trials=20,
        study_name='yolo-optimization',
        storage=None  # Use 'sqlite:///optuna_study.db' for persistent storage
    )


if __name__ == '__main__':
    main()
