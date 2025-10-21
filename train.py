import sys
sys.path.append('../')

import functools
from typing import NamedTuple, Any
import time

import jax
import jax.numpy as jnp

from jaxopt._src import base, tree_util
from jaxopt import OptaxSolver
import optax

from models.resnet import ResNet1, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from config.optimizer import get_velo_optimizer
from utils import save_checkpoint
from jeffnet.linen import create_model #effnet

class OptaxState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  error: float
  internal_state: NamedTuple
  aux: Any

# we need to reimplement optax's OptaxSolver's lopt_update method to properly pass in the loss data that VeLO expects.
def lopt_update(self,
            params: Any,
            state: NamedTuple,
            *args,
            **kwargs) -> base.OptStep:
  """Performs one iteration of the optax solver.

  Args:
    params: pytree containing the parameters.
    state: named tuple containing the solver state.
    *args: additional positional arguments to be passed to ``fun``.
    **kwargs: additional keyword arguments to be passed to ``fun``.
  Returns:
    (params, state)
  """
  if self.pre_update:
    params, state = self.pre_update(params, state, *args, **kwargs)

  (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)

  # note the only difference between this function and the baseline 
  # optax.OptaxSolver.lopt_update is that `extra_args` is now passed.
  # if you would like to use a different optimizer, you will likely need to
  # remove these extra_args.

  # delta is like the -learning_rate * grad, though more complex in VeLO (detail in paper)
  delta, opt_state = self.opt.update(
    grad, state.internal_state, params, extra_args={"loss": value}
  )
  # applies the actual update to parameters
  params = self._apply_updates(params, delta)

  # Computes optimality error before update to re-use grad evaluation.
  new_state = OptaxState(iter_num=state.iter_num + 1,
                          error=tree_util.tree_l2_norm(grad),
                          value=value,
                          aux=aux,
                          internal_state=opt_state)
  # return both updated params and training state used in the next iteration of training
  return base.OptStep(params=params, state=new_state)

from flax.core import frozen_dict

def count_parameters(params):
    """Counts total number of trainable parameters in a JAX model."""
    leaves = jax.tree_util.tree_leaves(frozen_dict.unfreeze(params))
    return sum(x.size for x in leaves)

########################################## main training ############################################
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

def train(train_ds, test_ds, ds_info, model, hparams: dict, 
          rng=None, opt='adam', init_params=None, init_batch_stats=None,
          early_stopping=False, patience=15, min_delta=0.0001,
          nb_epochs=10, train_batch_size=32, test_batch_size=32):
    
    print(f"################### TRAINING WITH {opt} ###################")

    input_shape = (1,) + ds_info.features["image"].shape
    num_classes = ds_info.features["label"].num_classes
    model = model.lower()
    opt = opt.lower()

    print(num_classes)
    
    if model == "efficientnet":
        net, init_variables = create_model(
            variant="tf_efficientnet_b0",
            pretrained=True,
            input_shape=(3, 32, 32),  # CIFAR100 shape RGB reverse
            num_classes=10
        )
        init_params = init_variables["params"]
        init_batch_stats = init_variables.get("batch_stats", {})
        
    else:
        net = {
            'resnet1': ResNet1,
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'resnet152': ResNet152
        }.get(model, None)

        if net is None:
            raise ValueError("Unknown model!")
        net = net(num_classes=num_classes)

    def predict(params, inputs, aux, train=False):
        all_params = {"params": params, "batch_stats": aux}
        if model == "efficientnet":
            return net.apply(all_params, inputs, training=train, mutable=["batch_stats"] if train else False)
        else:
            return net.apply(all_params, inputs, train=train, mutable=["batch_stats"] if train else False)
            
    @jax.jit
    def compute_softmax_loss(params, logits, labels):
        """Loss for soft labels (CutMix / MixUp compatible)."""
        loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        sqnorm = tree_util.tree_l2_norm(params, squared=True)
        return loss_val 
        
    @jax.jit
    def accuracy_and_loss(params, data, aux):
        inputs, labels = data
        logits = predict(params, inputs, aux, train=False)
        pred_class = jnp.argmax(logits, axis=-1)
        true_class = jnp.argmax(labels, axis=-1)  # Convert soft labels back to class
        acc = jnp.mean(pred_class == true_class)
        loss_val = compute_softmax_loss(params, logits, labels)
        return acc, loss_val

    def loss_fun(params, data, aux):
        inputs, labels = data
        logits, net_state = predict(params, inputs, aux, train=True)
        loss_val = compute_softmax_loss(params, logits, labels)
        return loss_val, net_state["batch_stats"]
    
    iter_per_epoch_train = ds_info.splits["train[:90%]"].num_examples // train_batch_size
    iter_per_epoch_test = ds_info.splits["train[90%:]"].num_examples // test_batch_size
    NUM_STEPS = nb_epochs * iter_per_epoch_train

    LR = hparams['lr']
    MOMENTUM = hparams['momentum']
    
    base_opt = (
        get_velo_optimizer(NUM_STEPS) if opt == 'velo'
        else optax.sgd(learning_rate=LR) if opt == 'sgd'
        else optax.sgd(learning_rate=LR, momentum=MOMENTUM) if opt == 'sgd_momentum' 
        else optax.adam(LR) if opt == 'adam'
        else optax.adamw(LR) 
    )

    solver = OptaxSolver(
        opt=base_opt,
        fun=jax.value_and_grad(loss_fun, has_aux=True),
        maxiter=NUM_STEPS,
        has_aux=True,
        value_and_grad=True,
    )

    if init_params is None:
        init_vars = net.init(rng, jnp.zeros(input_shape), train=True)
        init_params = init_vars["params"]
        init_batch_stats = init_vars["batch_stats"]

    params = init_params
    batch_stats = init_batch_stats
    NUM_PARAMS = count_parameters(params)
    
    state = solver.init_state(params, next(test_ds), batch_stats)

    jitted_update = jax.jit(functools.partial(lopt_update, solver)) if opt == 'velo' else jax.jit(solver.update)
    
    
    print(f"\nnumber of parameters for {model}: {NUM_PARAMS}\n")
    
    # tensorboard logs writer
    writer = SummaryWriter(log_dir=f"runs/{ds_info.name}/{model}_{opt}")
        
    print(f"Training {NUM_STEPS} steps over {nb_epochs} epochs")

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    epoch_times = []
    step_train_acc, step_train_loss = [], []

    total_start = time.time()

    best_val_metric = None
    best_params = None
    best_batch_stats = None
    patience_counter = 0
    best_epoch = 0
    
    step_counter = 0
    test_data = list(test_ds)

    for epoch in trange(nb_epochs, desc="Epochs"):
        epoch_start = time.time()
        pbar = tqdm(range(iter_per_epoch_train), leave=False, desc="Steps")

        for step in pbar:
            train_minibatch = next(train_ds)
            acc_train, loss_train = accuracy_and_loss(params, train_minibatch, batch_stats)
            
            # log step-level metrics to TensorBoard
            step_counter = epoch * iter_per_epoch_train + step
            writer.add_scalar("Step/accuracy", float(acc_train), step_counter)
            writer.add_scalar("Step/loss", float(loss_train), step_counter)
            
            # train step logging
            step_train_acc.append(float(acc_train))
            step_train_loss.append(float(loss_train))
            
            # update params optimizer step()
            params, state = jitted_update(params=params, state=state, data=train_minibatch, aux=batch_stats)
            
            batch_stats = state.aux
            pbar.set_postfix(acc=acc_train.item(), loss=loss_train.item())

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # calculate train metrics
        train_eval_batches = [next(train_ds) for _ in range(iter_per_epoch_train)]
        train_metrics = [accuracy_and_loss(params, batch, batch_stats) for batch in train_eval_batches]
        train_acc = float(jnp.mean(jnp.array([m[0] for m in train_metrics])))
        train_loss = float(jnp.mean(jnp.array([m[1] for m in train_metrics])))

        # caculate validation metrics
        #test_batches = [next(test_ds) for _ in range(iter_per_epoch_test)]
        #test_metrics = [accuracy_and_loss(params, batch, batch_stats) for batch in test_batches]
        test_metrics = [accuracy_and_loss(params, batch, batch_stats) for batch in test_data]
        test_acc = float(jnp.mean(jnp.array([m[0] for m in test_metrics])))
        test_loss = float(jnp.mean(jnp.array([m[1] for m in test_metrics])))

        # log to tensorboard 
        writer.add_scalar("Train/accuracy", train_acc, epoch)
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Test/accuracy", test_acc, epoch)
        writer.add_scalar("Test/loss", test_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        
        tqdm.write(f"[Epoch {epoch+1}] Train acc: {train_acc:.3f}, loss: {train_loss:.3f} | "
                   f"Test acc: {test_acc:.3f}, loss: {test_loss:.3f} | Time: {epoch_time:.2f}s")

        
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        current_val_metric = test_acc  # or test_loss
        improved = best_val_metric is None or (current_val_metric > best_val_metric + min_delta)

        if improved:
            best_val_metric = current_val_metric
            best_epoch = epoch
            patience_counter = 0
            best_params = params
            best_batch_stats = batch_stats
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}")
                params = best_params
                batch_stats = best_batch_stats
                break

    writer.close()
    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.2f}s")

    #####
    #filename = f"{opt}_{model}_{ds_info.name}_pretrained.msgpack"
    #save_checkpoint(params, batch_stats, NUM_STEPS, path=f'results/checkpoints/{ds_info.name}/{filename}') 

    return {
        "train_acc": train_acc_list,
        "train_loss": train_loss_list,
        "test_acc": test_acc_list,
        "test_loss": test_loss_list,
        "step_train_acc": step_train_acc,
        "step_train_loss": step_train_loss,
        "epoch_times": epoch_times,
        "total_training_time": total_time,
        "nb_steps": NUM_STEPS,
        "params": params,
        "batch_stats": batch_stats
    }

