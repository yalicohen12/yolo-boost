"""
Configuration presets for common YOLO optimization scenarios.
"""

PRESETS = {
    'quick': {
        'description': 'Quick experiment - fast iterations for testing',
        'n_trials': 3,
        'epochs': 10,
        'patience': 5,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo12n.pt', 'yolo12s.pt'],
        'optimization_metric': 'mAP50',
        'batch_options': '16,32',
        'search_params': 'model_version,optimizer,batch,lr0,lrf,momentum,weight_decay,box,cls,mosaic',
        # imgsz is auto-detected from dataset
    },

    'accuracy': {
        'description': 'Accuracy-focused - maximize mAP with larger models',
        'n_trials': 30,
        'epochs': 50,
        'patience': 20,
        'models': ['yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': '8,16',
        # imgsz is auto-detected from dataset
    },

    'speed': {
        'description': 'Speed-focused - optimize for fast inference',
        'n_trials': 20,
        'epochs': 30,
        'patience': 10,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo12n.pt', 'yolo12s.pt'],
        'optimization_metric': 'speed',
        'batch_options': '16,32,64',
        # imgsz is auto-detected from dataset
    },

    'balanced': {
        'description': 'Balanced - good accuracy with reasonable speed',
        'n_trials': 25,
        'epochs': 40,
        'patience': 15,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt'],
        'optimization_metric': 'balanced',
        'batch_options': '8,16,32',
        # imgsz is auto-detected from dataset
    },

    'production': {
        'description': 'Production - thorough search for deployment',
        'n_trials': 50,
        'epochs': 100,
        'patience': 50,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt',
                   'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': '8,16,32',
        # imgsz is auto-detected from dataset
    },

    'focused': {
        'description': 'Focused - searches only the 10 highest-impact params for better TPE convergence',
        'n_trials': 30,
        'epochs': 50,
        'patience': 20,
        'models': ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt'],
        'optimization_metric': 'mAP50-95',
        'batch_options': '8,16,32',
        # Only these params are searched — everything else uses YOLO11/12 defaults.
        # 10 params vs 28 means TPE gets ~3x better coverage per trial.
        'search_params': 'model_version,optimizer,batch,lr0,lrf,momentum,weight_decay,box,cls,mosaic',
        # imgsz is auto-detected from dataset
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
