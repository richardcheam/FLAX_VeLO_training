"""Factory for a VeLO optimiser wrapped as an Optax optimiser."""
from learned_optimization.research.general_lopt import prefab
import optax

def get_velo_optimizer(num_steps: int = 5000):
    """Return a pre‑trained VeLO optimiser for the given number of steps."""
    return prefab.optax_lopt(num_steps)

def build_optimizer(hparams: dict, num_steps, use_lopt, alternate_opt, momentum):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=hparams["lr_init"],
        peak_value=hparams["lr_peak"],
        warmup_steps=num_steps // 10,
        decay_steps=num_steps,
        end_value=hparams["lr_end"]
    )
    base_opt = (
        get_velo_optimizer(num_steps) if use_lopt
        else optax.sgd(learning_rate=schedule) if alternate_opt == 'sgd'
        else optax.sgd(learning_rate=schedule, momentum=momentum) if alternate_opt == 'sgd_momentum' 
        else optax.adam(schedule) if alternate_opt == 'adam'
        else optax.adamw(schedule) 
    )
    return optax.chain(
        optax.clip_by_global_norm(hparams["clip_threshold"]),
        base_opt
    )
