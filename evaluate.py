# evaluate.py
import jax, jax.numpy as jnp
from jaxopt import loss
from flax.training.train_state import TrainState
from jaxopt._src import tree_util           # for the L2-reg term
from typing import Any
from models.resnet import ResNet1, ResNet18, ResNet34
import optax

# inherit TrainState
class EvalState(TrainState):
    batch_stats: Any
    #l2reg: float #add any state for inference, here l2reg is not use anymore

@jax.jit
def _logistic_loss(logits, labels):
    """Cross-entropy"""
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    return loss_val

# def evaluate(state: EvalState, test_ld, iter_per_epoch_test):
#     """Returns (accuracy, loss)"""
#     @jax.jit
#     def eval_step(params, batch_stats, batch):
#         inputs, labels = batch

#         variables = {"params": params, "batch_stats": batch_stats}
#         logits = state.apply_fn(variables, inputs, train=False, mutable=False)
  
#         pred_class = jnp.argmax(logits, axis=-1) 
#         true_class = jnp.argmax(labels, axis=-1)  # Convert soft labels back to class
        
#         acc = jnp.mean(pred_class == true_class)

#         loss = _logistic_loss(params, logits, labels)
#         return acc, loss
    
#     test_batches = [next(iter(test_ld)) for _ in range(iter_per_epoch_test)]
#     test_metrics = [eval_step(state.params, state.batch_stats, state.l2reg, batch) for batch in test_batches]
#     test_acc = float(jnp.mean(jnp.array([m[0] for m in test_metrics])))
#     test_loss = float(jnp.mean(jnp.array([m[1] for m in test_metrics])))

#     print(f"acc={test_acc:.4f}  loss={test_loss:.4f}")
#     return test_acc, test_loss

def evaluate(state: EvalState, test_iter):
    """Returns (accuracy, loss)"""

    @jax.jit
    def eval_step(params, batch_stats, batch):
        inputs, labels = batch

        variables = {"params": params, "batch_stats": batch_stats}
        logits = state.apply_fn(variables, inputs, train=False, mutable=False)

        pred_class = jnp.argmax(logits, axis=-1)
        true_class = jnp.argmax(labels, axis=-1)

        acc = jnp.mean(pred_class == true_class)
        loss = _logistic_loss(logits, labels)
        return acc, loss

    accs, losses = [], []

    for batch in list(test_iter):  # no need to count, just go through once
        acc, loss = eval_step(state.params, state.batch_stats, batch)
        accs.append(acc)
        losses.append(loss)

    test_acc = float(jnp.mean(jnp.array(accs)))
    test_loss = float(jnp.mean(jnp.array(losses)))

    print(f"acc={test_acc:.4f}  loss={test_loss:.4f}")
    return test_acc, test_loss