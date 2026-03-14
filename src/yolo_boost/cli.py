"""
yolo-boost CLI entry point.
"""

from yolo_boost.trainer import YOLOOptunaTrainer
from yolo_boost.presets import get_preset, list_presets, PRESETS
import argparse
import os
import shutil
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

CONFIG_FILE = '.yolo-boost-config'


def cmd_init(args):
    """Write a config template to the current directory."""
    dest = Path(CONFIG_FILE)
    if dest.exists() and not args.force:
        console.print(f"[yellow]{dest}[/yellow] already exists. Use [bold]--force[/bold] to overwrite.")
        return

    template = Path(__file__).parent / CONFIG_FILE
    shutil.copy(template, dest)
    console.print(f"[green]Created[/green] {dest}")
    console.print(f"Edit it with your settings, then run [cyan]yolo-boost run --preset quick --data data.yaml[/cyan]")


def cmd_run(args):
    """Run a hyperparameter optimization study or baseline training."""
    # Warn if running with no preset and no config file — defaults are very aggressive
    if not args.preset and not Path(CONFIG_FILE).exists():
        console.print(Panel(
            f"[yellow]No preset and no {CONFIG_FILE} found.[/yellow]\n"
            "Running with defaults: [bold]20 trials × 50 epochs[/bold], many model sizes.\n"
            "This will take a long time.\n\n"
            "[bold]Quick start options:[/bold]\n"
            "  [cyan]yolo-boost run --preset quick --data data.yaml[/cyan]\n"
            f"  [cyan]yolo-boost init  # then edit {CONFIG_FILE}[/cyan]",
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
        ))
        console.print()

    # Apply preset if specified
    if args.preset:
        preset = get_preset(args.preset)
        console.print(f"\n[bold]Using preset:[/bold] [cyan]{args.preset}[/cyan]")
        console.print(f"[dim]{preset['description']}[/dim]\n")

        if args.trials is None:
            args.trials = preset['n_trials']
        if args.models is None:
            env_models = os.getenv('MODEL_VERSIONS')
            args.models = [m.strip() for m in env_models.split(',')] if env_models else preset['models']
        if args.optimization_metric is None:
            args.optimization_metric = os.getenv('OPTIMIZATION_METRIC') or preset['optimization_metric']

        # env file > preset > hardcoded default — only fill gaps the env didn't cover
        os.environ.setdefault('EPOCHS', str(preset['epochs']))
        os.environ.setdefault('PATIENCE', str(preset.get('patience', '50')))
        os.environ.setdefault('BATCH_OPTIONS', preset.get('batch_options', '8,16,32'))
        if preset.get('search_params'):
            os.environ.setdefault('SEARCH_PARAMS', preset['search_params'])

    # CLI --epochs / --patience always win over everything
    if args.epochs is not None:
        os.environ['EPOCHS'] = str(args.epochs)
    if args.patience is not None:
        os.environ['PATIENCE'] = str(args.patience)

    if args.trials is None:
        args.trials = int(os.getenv('N_TRIALS', '20'))

    trainer = YOLOOptunaTrainer(
        data_yaml=args.data,
        mlflow_tracking_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        model_versions=args.models,
        device=args.device,
        optimization_metric=args.optimization_metric,
        run_name=args.run_name
    )

    # Baseline mode
    if args.baseline:
        table = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
        table.add_column(style="bold cyan", no_wrap=True, min_width=14)
        table.add_column()
        if args.preset:
            table.add_row("Preset", args.preset)
        table.add_row("Run Name", trainer.run_name)
        table.add_row("Model", args.model if args.model else "from config (first in list)")
        table.add_row("Data", args.data)
        table.add_row("Epochs", str(args.epochs) if args.epochs else "from config")
        table.add_row("MLflow URI", trainer.mlflow_tracking_uri)
        table.add_row("Experiment", str(args.experiment))
        table.add_row("Device", args.device if args.device else "from config")
        console.print(Panel(table, title="[bold blue] YOLO Baseline Training [/bold blue]", border_style="blue"))
        console.print(f"\nResults will be saved to: [cyan]runs/optuna/{trainer.run_name}/[/cyan]")

        if args.dry_run:
            console.print(Panel(
                f"[green]Config and dataset look good.[/green]\n"
                f"Would run [bold]1 baseline trial[/bold] × [bold]{trainer.epochs} epochs[/bold] "
                f"on [bold]{args.model or trainer.model_versions[0]}[/bold].\n\n"
                "[dim]Remove --dry-run to start training.[/dim]",
                title="[bold green] Dry Run Complete [/bold green]",
                border_style="green",
            ))
            return

        trainer.train_baseline(model_name=args.model, epochs=args.epochs)

        console.print("\n[bold green]Baseline training complete![/bold green]")
        console.print(f"MLflow tracking: [cyan]{trainer.mlflow_tracking_uri}[/cyan]")
        console.print('Run [bold]mlflow ui[/bold] to view results in browser.')
        return

    # Optimization mode — print all resolved config before starting
    search_params = ', '.join(sorted(trainer.search_params)) if trainer.search_params else 'all'

    table = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
    table.add_column(style="bold cyan", no_wrap=True, min_width=16)
    table.add_column()
    table.add_row("Preset", args.preset or "[dim]none[/dim]")
    table.add_row("Run name", trainer.run_name)
    table.add_row("Study name", args.study_name)
    table.add_row("Storage", args.storage or "[dim]in-memory (not persistent)[/dim]")
    table.add_section()
    table.add_row("Data", trainer.data_yaml)
    table.add_row("Device", trainer.device)
    table.add_row("Image size", f"{trainer.imgsz}  [dim](auto-detected from dataset)[/dim]")
    table.add_section()
    table.add_row("Trials", str(args.trials))
    table.add_row("Epochs/trial", str(trainer.epochs))
    table.add_row("Patience", str(trainer.patience))
    table.add_row("Optimize for", trainer.optimization_metric)
    table.add_section()
    table.add_row("Models", ", ".join(trainer.model_versions))
    table.add_row("Optimizers", ", ".join(trainer.optimizer_options))
    table.add_row("Batch options", ", ".join(str(b) for b in trainer.batch_options))
    table.add_row("Search params", search_params)
    table.add_section()
    table.add_row("MLflow URI", trainer.mlflow_tracking_uri)
    table.add_row("Experiment", trainer.experiment_name)
    table.add_row("Output dir", f"runs/optuna/{trainer.run_name}/")
    console.print(Panel(table, title="[bold blue] YOLO Boost — Hyperparameter Optimization [/bold blue]", border_style="blue"))

    if args.dry_run:
        console.print(Panel(
            f"[green]Config and dataset look good.[/green]\n"
            f"Would run [bold]{args.trials} trials[/bold] × [bold]{trainer.epochs} epochs[/bold] "
            f"across [bold]{len(trainer.model_versions)} model(s)[/bold], "
            f"optimizing for [bold]{trainer.optimization_metric}[/bold].\n\n"
            "[dim]Remove --dry-run to start training.[/dim]",
            title="[bold green] Dry Run Complete [/bold green]",
            border_style="green",
        ))
        return

    time.sleep(2)

    study = trainer.optimize(
        n_trials=args.trials,
        study_name=args.study_name,
        storage=args.storage
    )

    console.print("\n[bold green]Optimization complete![/bold green]")
    console.print(f"MLflow tracking: [cyan]{trainer.mlflow_tracking_uri}[/cyan]")
    console.print('Run [bold]mlflow ui[/bold] to view results in browser.')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='yolo-boost',
        description=(
            'Automated hyperparameter optimization for YOLO models.\n'
            'Uses Optuna (TPE) to search hyperparameters across multiple trials\n'
            'and logs everything to MLflow for easy comparison.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
New to yolo-boost? Start here:
  1. Run `yolo-boost init` in your project directory.
     This creates a .yolo-boost-config file that documents every config
     option (MLflow URI, device, model sizes, search ranges, etc.).
     Edit it with your settings.

  2. Run `yolo-boost run --preset quick --data your_data.yaml`
     Presets are the easiest way to get started — they set sensible values
     for trials, epochs, models, and which params to search.
     Available presets: quick, accuracy, speed, balanced, focused, production

  3. Run `mlflow ui` to explore results in your browser.
     All trial metrics, hyperparameters, and plots are tracked automatically.

Example (first time):
  cd my-project/
  yolo-boost init
  yolo-boost run --preset quick --data data.yaml
  mlflow ui
""",
    )
    subparsers = parser.add_subparsers(dest='command')

    # ── init ──────────────────────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        'init',
        help=f'Create a {CONFIG_FILE} config file in the current directory',
        description=(
            f'Write a {CONFIG_FILE} template to the current directory.\n'
            'It documents every config option with sensible defaults.\n'
            'Edit it with your settings and you\'re ready to go.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_parser.add_argument('--force', action='store_true', help=f'Overwrite existing {CONFIG_FILE}')

    # ── run ───────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        'run',
        help='Run optimization (or baseline training)',
        description=(
            'Run Optuna hyperparameter optimization across multiple YOLO models.\n'
            'Each trial trains YOLO with a different set of hyperparameters and\n'
            'reports results to MLflow. Use a preset for sensible defaults, or\n'
            'pass individual flags to override anything.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets:
  {', '.join(PRESETS.keys())}

Use --list-presets to see detailed preset descriptions.

Examples:
  yolo-boost run --preset quick --data my_data.yaml
  yolo-boost run --preset accuracy --device 0
  yolo-boost run --data data.yaml --trials 30 --device cuda:0
  yolo-boost run --baseline --data data.yaml
        """,
    )
    run_parser.add_argument('--preset', type=str, default=None,
                            help='Use a configuration preset (quick, accuracy, speed, balanced, production)')
    run_parser.add_argument('--list-presets', action='store_true',
                            help='List all available presets and exit')
    run_parser.add_argument('--baseline', action='store_true',
                            help='Run single baseline training (no optimization) for comparison')
    run_parser.add_argument('--models', type=str, nargs='+', default=None,
                            help='YOLO models to optimize (e.g., yolo11n.pt yolo11s.pt)')
    run_parser.add_argument('--model', type=str, default=None,
                            help='Single model for baseline training (e.g., yolo11n.pt)')
    run_parser.add_argument('--data', type=str, default='data.yaml',
                            help='Path to data.yaml configuration file')
    run_parser.add_argument('--trials', type=int, default=None,
                            help='Number of optimization trials')
    run_parser.add_argument('--epochs', type=int, default=None,
                            help='Number of epochs per trial')
    run_parser.add_argument('--patience', type=int, default=None,
                            help='Early stopping patience')
    run_parser.add_argument('--mlflow-uri', type=str, default=None,
                            help='MLflow tracking URI (default: ./mlruns)')
    run_parser.add_argument('--experiment', type=str, default=None,
                            help='MLflow experiment name')
    run_parser.add_argument('--study-name', type=str, default='yolo-optimization',
                            help='Optuna study name')
    run_parser.add_argument('--storage', type=str, default=None,
                            help='Optuna storage (e.g., sqlite:///optuna_study.db)')
    run_parser.add_argument('--device', type=str, default=None,
                            help='Device (cpu, 0, 1, 2, ...)')
    run_parser.add_argument('--optimization-metric', type=str, default=None,
                            help='Metric to optimize (mAP50, mAP50-95, speed, balanced)')
    run_parser.add_argument('--run-name', type=str, default=None,
                            help='Custom run name (default: auto-generated timestamp)')
    run_parser.add_argument('--dry-run', action='store_true',
                            help='Validate config and dataset, print resolved settings, then exit without training')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'init':
        cmd_init(args)
    elif args.command == 'run':
        if args.list_presets:
            list_presets()
            return
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
