# HYPERPARAMETER CONFIGURATION 
# space search: uniform behavior from specified interval
def suggest_hparams(trial):
    return {
        # Learning rate warmup + cosine schedule range
        "lr_init": trial.suggest_loguniform("lr_init", 1e-5, 1e-3),   # Lower bound for stable warmup
        "lr_peak": trial.suggest_loguniform("lr_peak", 1e-3, 3e-2),   # Peak LR (higher for SGD, lower for Adam)
        "lr_end": trial.suggest_loguniform("lr_end", 1e-6, 1e-4),     # Final LR at end of cosine decay

        # Gradient clipping
        "clip_threshold": trial.suggest_float("clip_threshold", 0.1, 1.0),  # VeLO is sensitive to clipping

        # Weight decay (L2 regularization)
        "l2reg": trial.suggest_loguniform("l2reg", 1e-6, 1e-4),  # Smaller values are safer for Adam

        # EMA for validation smoothing
        "ema_decay": trial.suggest_float("ema_decay", 0.95, 0.999, step=0.005),

        # Optimizer momentum (SGD variants)
        #"momentum": trial.suggest_float("momentum", 0.85, 0.99, step=0.05),  # More useful for SGD than Adam
    }


def log_hyperparams(writer, hparams):
    """Logs hyperparameters to TensorBoard"""
    for key, val in hparams.items():
        writer.add_text("hyperparams/" + key, str(val))
