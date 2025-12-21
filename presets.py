"""
Configuration presets for common YOLO optimization scenarios.
"""

PRESETS = {
    'quick': {
        'description': 'Quick experiment - fast iterations for testing',
        'n_trials': 5,
        'epochs': 10,
        'models': ['yolo11n.pt', 'yolo11s.pt'],
        'optimization_metric': 'mAP50',
        'batch_options': '16,32',
        'imgsz_options': '320,416',
    },

    'accuracy': {
        'description': 'Accuracy-focused - maximize mAP with larger models',
        'n_trials': 30,
        'epochs': 50,
        'models': ['yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': '8,16',
        'imgsz_options': '512,640',
    },

    'speed': {
        'description': 'Speed-focused - optimize for fast inference',
        'n_trials': 20,
        'epochs': 30,
        'models': ['yolo11n.pt', 'yolo11s.pt'],
        'optimization_metric': 'speed',
        'batch_options': '16,32,64',
        'imgsz_options': '320,416,512',
    },

    'balanced': {
        'description': 'Balanced - good accuracy with reasonable speed',
        'n_trials': 25,
        'epochs': 40,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'],
        'optimization_metric': 'balanced',
        'batch_options': '8,16,32',
        'imgsz_options': '416,512,640',
    },

    'production': {
        'description': 'Production - thorough search for deployment',
        'n_trials': 50,
        'epochs': 100,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': '8,16,32',
        'imgsz_options': '416,512,640',
    },
}


def get_preset(preset_name):
    """Get preset configuration by name."""
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    return PRESETS[preset_name]


def list_presets():
    """List all available presets with descriptions."""
    print("\nAvailable presets:\n")
    for name, config in PRESETS.items():
        print(f"  {name:12} - {config['description']}")
        print(f"               Trials: {config['n_trials']}, Epochs: {config['epochs']}, "
              f"Models: {', '.join(config['models'])}")
        print()
