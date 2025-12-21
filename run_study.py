#!/usr/bin/env python3
"""
Example script to run YOLO Optuna optimization study.
Customize the parameters below for your use case.
"""

from train_yolo_optuna import YOLOOptunaTrainer
from presets import get_preset, list_presets, PRESETS
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run YOLO Optuna optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets:
  {', '.join(PRESETS.keys())}

Use --list-presets to see detailed preset descriptions.

Examples:
  python run_study.py --preset quick --data my_data.yaml
  python run_study.py --preset accuracy --device 0
  python run_study.py --data data.yaml --trials 30 --device cuda:0
        """
    )
    parser.add_argument('--preset', type=str, default=None,
                        help='Use a configuration preset (quick, accuracy, speed, balanced, production)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all available presets and exit')
    parser.add_argument('--baseline', action='store_true',
                        help='Run single baseline training (no optimization) for comparison')
    parser.add_argument('--models', type=str, nargs='+',
                        default=None,  # Use .env MODEL_VERSIONS instead
                        help='YOLO models to optimize (e.g., yolo11n.pt yolo11s.pt)')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model for baseline training (e.g., yolo11n.pt)')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml configuration file')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs per trial')
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5000',
                        help='MLflow tracking URI')
    parser.add_argument('--experiment', type=str, default=None,
                        help='MLflow experiment name (default: same as study-name)')
    parser.add_argument('--study-name', type=str, default='yolo-optimization',
                        help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage (e.g., sqlite:///optuna_study.db)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cpu, 0, 1, 2, etc.)')
    parser.add_argument('--optimization-metric', type=str, default=None,
                        help='Metric to optimize (mAP50, mAP50-95, speed, balanced)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name (default: auto-generated timestamp)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        return

    # Apply preset if specified
    if args.preset:
        preset = get_preset(args.preset)
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {preset['description']}\n")

        # Apply preset values (command-line args override preset)
        if args.trials is None:
            args.trials = preset['n_trials']
        if args.models is None:
            args.models = preset['models']
        if args.optimization_metric is None:
            args.optimization_metric = preset['optimization_metric']
        if args.epochs is None:
            args.epochs = preset['epochs']

        # Set env vars for preset (will be used by trainer if not overridden)
        os.environ['BATCH_OPTIONS'] = preset.get('batch_options', '8,16,32')
        os.environ['IMGSZ_OPTIONS'] = preset.get('imgsz_options', '320,416,512,640')
        os.environ['EPOCHS_RANGE'] = f"{preset['epochs']},{preset['epochs']}"

    # Set defaults if still None
    if args.trials is None:
        args.trials = 20

    # Use study name as experiment name if not specified
    if args.experiment is None:
        args.experiment = args.study_name

    # Create trainer first to get run_name
    trainer = YOLOOptunaTrainer(
        data_yaml=args.data,
        mlflow_tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        model_versions=args.models,
        device=args.device,
        optimization_metric=args.optimization_metric,
        run_name=args.run_name
    )

    # Baseline mode - single training run
    if args.baseline:
        print('='*70)
        print('YOLO Baseline Training (Single Run - No Optimization)')
        print('='*70)
        if args.preset:
            print(f'Preset: {args.preset}')
        print(f'Run Name: {trainer.run_name}')
        print(f'Model: {args.model if args.model else "from .env (first in list)"}')
        print(f'Data: {args.data}')
        print(f'Epochs: {args.epochs if args.epochs else "from .env"}')
        print(f'MLflow URI: {args.mlflow_uri}')
        print(f'Experiment: {args.experiment}')
        print(f'Device: {args.device if args.device else "from .env"}')
        print('='*70)
        print(f'\nResults will be saved to: runs/optuna/{trainer.run_name}/')
        print('Note: This is a baseline run for comparison, not optimization!')
        print('='*70)

        trainer.train_baseline(
            model_name=args.model,
            epochs=args.epochs
        )

        print('\nBaseline training complete!')
        print(f'View results in MLflow UI at: {args.mlflow_uri}')
        return

    # Regular optimization mode
    print('='*70)
    print('YOLO Training with Optuna Hyperparameter Optimization')
    print('='*70)
    if args.preset:
        print(f'Preset: {args.preset}')
    print(f'Run Name: {trainer.run_name}')
    print(f'Models to optimize: {", ".join(args.models) if args.models else "from .env"}')
    print(f'Data: {args.data}')
    print(f'Trials: {args.trials}')
    print(f'Epochs per trial: {args.epochs if args.epochs else "from .env"}')
    print(f'Optimization metric: {args.optimization_metric if args.optimization_metric else "from .env"}')
    print(f'MLflow URI: {args.mlflow_uri}')
    print(f'Experiment: {args.experiment}')
    print(f'Study Name: {args.study_name}')
    print(f'Storage: {args.storage}')
    print(f'Device: {args.device if args.device else "from .env"}')
    print('='*70)
    print(f'\nResults will be saved to: runs/optuna/{trainer.run_name}/')
    print('Note: Model version/size is auto-tuned by Optuna!')
    print('='*70)

    study = trainer.optimize(
        n_trials=args.trials,
        study_name=args.study_name,
        storage=args.storage
    )

    print('\nOptimization complete!')
    print(f'View results in MLflow UI at: {args.mlflow_uri}')


if __name__ == '__main__':
    main()
