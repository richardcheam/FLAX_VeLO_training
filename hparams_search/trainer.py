import sys
sys.path.append('../')

import functools
from typing import NamedTuple, Any
import time

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from jaxopt._src import base, tree_util
from jaxopt import OptaxSolver
from jaxopt import loss

import functools
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.numpy as jnp
from jaxopt._src import base
from jaxopt._src import tree_util
import optax

from models.resnet import ResNet1, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from config.hparams import log_hyperparams

from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

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

########################################## main training ############################################
def trainer(train_ds, test_ds, ds_info, model, hparams: dict, optimizer_fn,
          rng=None, use_lopt=True, init_params=None, init_batch_stats=None,
          swa_start_epoch=30, swa_freq=3,
          alternate_opt='adam', nb_epochs=10, train_batch_size=32, test_batch_size=32,
          run_name="default"):

    input_shape = (1,) + ds_info.features["image"].shape
    num_classes = ds_info.features["label"].num_classes
    # convert model and optimizer name
    model = model.lower()
    alternate_opt = alternate_opt.lower()

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
        x = inputs.astype(jnp.float32) / 255.
        all_params = {"params": params, "batch_stats": aux}
        return net.apply(all_params, x, train=train, mutable=["batch_stats"] if train else False)

    @jax.jit
    def compute_softmax_loss(params, l2reg, logits, labels):
        """Loss for soft labels (CutMix / MixUp compatible)."""
        loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        sqnorm = tree_util.tree_l2_norm(params, squared=True)
        return loss_val + 0.5 * l2reg * sqnorm
        
    @jax.jit
    def accuracy_and_loss(params, l2reg, data, aux):
        inputs, labels = data
        logits = predict(params, inputs, aux, train=False)
        pred_class = jnp.argmax(logits, axis=-1)
        true_class = jnp.argmax(labels, axis=-1)  # Convert soft labels back to class
        acc = jnp.mean(pred_class == true_class)
        loss_val = compute_softmax_loss(params, l2reg, logits, labels)
        return acc, loss_val

    def loss_fun(params, l2reg, data, aux):
        inputs, labels = data
        logits, net_state = predict(params, inputs, aux, train=True)
        loss_val = compute_softmax_loss(params, l2reg, logits, labels)
        return loss_val, net_state["batch_stats"]
        
    @jax.jit
    def update_ema(params, ema_state):
        return ema.update(params, ema_state)
    
    iter_per_epoch_train = ds_info.splits["train"].num_examples // train_batch_size
    iter_per_epoch_test = ds_info.splits["test"].num_examples // test_batch_size
    NUM_STEPS = nb_epochs * iter_per_epoch_train
    
    # DEFINE HPARAMS
    L2REG = hparams['l2reg']
    #LR = hparams['lr']
    #MOMENTUM = hparams['momentum']
    MOMENTUM=0.9
    EMA_DECAY = hparams['ema_decay']

    opt = optimizer_fn(hparams=hparams, num_steps=NUM_STEPS, use_lopt=use_lopt, alternate_opt=alternate_opt, momentum=MOMENTUM)
    
    solver = OptaxSolver(
        opt=opt,
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
    ## EMA
    ema = optax.ema(decay=EMA_DECAY)
    ema_state = ema.init(params)
    ## SWA
    swa_params = params
    swa_n = 0  # number of models averaged
    
    state = solver.init_state(params, L2REG, next(test_ds), batch_stats)
    jitted_update = jax.jit(functools.partial(lopt_update, self=solver)) if use_lopt else jax.jit(solver.update)

    # TENSORBOARD LOGGINGS #
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    log_hyperparams(writer, hparams)

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    epoch_times = []
    step_train_acc, step_train_loss = [], []

    total_start = time.time()

    for epoch in trange(nb_epochs, desc="Epochs"):
        epoch_start = time.time()
        pbar = tqdm(range(iter_per_epoch_train), leave=False, desc="Steps")

        for step in pbar:
            train_minibatch = next(train_ds)
            acc_train, loss_train = accuracy_and_loss(params, L2REG, train_minibatch, batch_stats)
            step_train_acc.append(float(acc_train))
            step_train_loss.append(float(loss_train))
            
            # log step-level metrics to TensorBoard
            step_counter = epoch * iter_per_epoch_train + step
            writer.add_scalar("Step/accuracy", float(acc_train), step_counter)
            writer.add_scalar("Step/loss", float(loss_train), step_counter)
            
            # UPDATE # 
            params, state = jitted_update(params=params, state=state, l2reg=L2REG, data=train_minibatch, aux=batch_stats)
            batch_stats = state.aux
            
            _, ema_state = update_ema(params, ema_state)
            if epoch >= swa_start_epoch and (epoch - swa_start_epoch) % swa_freq == 0:
                swa_params = jax.tree_util.tree_map(
                    lambda p1, p2: (p1 * swa_n + p2) / (swa_n + 1), swa_params, params
                )
            swa_n += 1
            ##########
            pbar.set_postfix(acc=acc_train.item(), loss=loss_train.item())

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        train_eval_batches = [next(train_ds) for _ in range(iter_per_epoch_train)]
        train_metrics = [accuracy_and_loss(params, L2REG, batch, batch_stats) for batch in train_eval_batches]
        train_acc = float(jnp.mean(jnp.array([m[0] for m in train_metrics])))
        train_loss = float(jnp.mean(jnp.array([m[1] for m in train_metrics])))

        test_batches = [next(test_ds) for _ in range(iter_per_epoch_test)]
        test_metrics = [accuracy_and_loss(params, L2REG, batch, batch_stats) for batch in test_batches]
        test_acc = float(jnp.mean(jnp.array([m[0] for m in test_metrics])))
        test_loss = float(jnp.mean(jnp.array([m[1] for m in test_metrics])))

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

    writer.close()
    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.2f}s")

    # EMA model
    ema_params = ema_state.ema
    ema_metrics = [accuracy_and_loss(ema_params, L2REG, batch, batch_stats) for batch in test_batches]
    ema_acc = float(jnp.mean(jnp.array([m[0] for m in ema_metrics])))
    ema_loss = float(jnp.mean(jnp.array([m[1] for m in ema_metrics])))
    
    swa_metrics = [accuracy_and_loss(swa_params, L2REG, batch, batch_stats) for batch in test_batches]
    swa_acc = float(jnp.mean(jnp.array([m[0] for m in swa_metrics])))
    swa_loss = float(jnp.mean(jnp.array([m[1] for m in swa_metrics])))

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
        "ema_params": ema_params,
        "swa_params": swa_params,
        "batch_stats": batch_stats,
        "l2reg": L2REG
    }
