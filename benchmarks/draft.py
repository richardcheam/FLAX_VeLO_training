import sys
sys.path.append('..')
from utils import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, Tuple

 
#tmp = load_seeded_metrics("results/metrics", "resnet1", "mnist", "velo")

#print(len(tmp[0]["train_acc"]))


import sys
sys.path.append('../')

from dataloader import TFDSDataLoader


train_loader = TFDSDataLoader('cifar10', split='train[:1%]', batch_size=32, is_training=True)
val_loader = TFDSDataLoader('cifar10', split='train[99%:]', batch_size=32, is_training=False)
test_loader = TFDSDataLoader('cifar10', split='test', batch_size=32, is_training=False)
ds_info = train_loader.get_info()
test_data = list(test_loader)

from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jeffnet.common import load_state_dict_from_url, split_state_dict, load_state_dict


import re
from typing import Any, Callable, Sequence, Dict
from functools import partial

from flax import linen as nn
import flax.linen.initializers as initializers
import jax
import jax.numpy as jnp

from jeffnet.common import round_features, get_model_cfg, EfficientNetBuilder
from jeffnet.linen.layers import conv2d, linear, batchnorm2d, get_act_fn
from jeffnet.linen.blocks_linen import ConvBnAct, SqueezeExcite, BlockFactory, Head, EfficientHead


ModuleDef = Any
Dtype = Any
effnet_normal = partial(initializers.variance_scaling, 2.0, "fan_out", "normal")
effnet_uniform = partial(initializers.variance_scaling, 1.0/3, "fan_out", "uniform")

class EfficientNet(nn.Module):
    """ EfficientNet (and other MBConvNets)
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-Lite
      * MixNet S, M, L, XL
      * MobileNetV3
      * MobileNetV2
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    # model config
    block_defs: Sequence[Sequence[Dict]]
    stem_size: int = 32
    feat_multiplier: float = 1.0
    feat_divisor: int = 8
    feat_min: int = None
    fix_stem: bool = False
    pad_type: str = 'LIKE'
    output_stride: int = 32

    # classifier / head config
    efficient_head: bool = False
    num_classes: int = 10
    num_features: int = 1280
    global_pool: str = 'avg'

    # pretrained / data config
    default_cfg: Dict = None

    # regularization
    drop_rate: float = 0.
    drop_path_rate: float = 0.

    dtype: Dtype = jnp.float32
    conv_layer: ModuleDef = conv2d
    norm_layer: ModuleDef = batchnorm2d
    se_layer: ModuleDef = SqueezeExcite
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool):
        # add dtype binding to layers
        # FIXME is there better way to handle dtype? Passing dtype to all child Modules also seems messy...
        lkwargs = dict(
            conv_layer=partial(self.conv_layer, dtype=self.dtype, kernel_init=effnet_normal()),
            norm_layer=partial(self.norm_layer, dtype=self.dtype),
            act_fn=self.act_fn)
        se_layer = partial(self.se_layer, dtype=self.dtype)
        linear_layer = partial(linear, dtype=self.dtype, kernel_init=effnet_uniform())

        stem_features = self.stem_size
        if not self.fix_stem:
            stem_features = round_features(self.stem_size, self.feat_multiplier, self.feat_divisor, self.feat_min)
        x = ConvBnAct(
            out_features=stem_features, kernel_size=3, stride=2, pad_type=self.pad_type,
            **lkwargs, name='stem')(x, training=training)

        blocks = EfficientNetBuilder(
            stem_features, self.block_defs, BlockFactory(),
            feat_multiplier=self.feat_multiplier, feat_divisor=self.feat_divisor, feat_min=self.feat_min,
            output_stride=self.output_stride, pad_type=self.pad_type, se_layer=se_layer, **lkwargs,
            drop_path_rate=self.drop_path_rate)()
        for stage in blocks:
            for block in stage:
                x = block(x, training=training)

        head_layer = EfficientHead if self.efficient_head else Head
        x = head_layer(
            num_features=self.num_features, num_classes=self.num_classes, drop_rate=self.drop_rate,
            **lkwargs, dtype=self.dtype, linear_layer=linear_layer, name='head')(x, training=training)
        return x


def _filter(state_dict):
    """ convert state dict keys from pytorch style origins to flax linen """
    out = {}
    p_blocks = re.compile(r'blocks\.(\d)\.(\d)')
    p_bn_scale = re.compile(r'bn(\w*)\.weight')
    for k, v in state_dict.items():
        k = p_blocks.sub(r'blocks_\1_\2', k)
        k = p_bn_scale.sub(r'bn\1.scale', k)
        k = k.replace('running_mean', 'mean')
        k = k.replace('running_var', 'var')
        k = k.replace('.weight', '.kernel')
        out[k] = v
    return out


def load_pretrained(variables, url='', default_cfg=None, filter_fn=None):
    if not url:
        assert default_cfg is not None and default_cfg['url']
        url = default_cfg['url']
    state_dict = load_state_dict_from_url(url, transpose=True)

    source_params, source_state = split_state_dict(state_dict)
    if filter_fn is not None:
        source_params = filter_fn(source_params)
        source_state = filter_fn(source_state)

    var_unfrozen = unfreeze(variables)
    missing_keys = []
    unexpected_keys = []

    # --- Handle parameters ---
    flat_params = flatten_dict(var_unfrozen['params'])
    flat_source_params = {
        tuple(k.split('.')): v for k, v in source_params.items()
    }

    for k, v in flat_params.items():
        if k in flat_source_params:
            source_val = flat_source_params[k]
            if v.shape == source_val.shape:
                flat_params[k] = source_val
            else:
                print(f"Skipping loading parameter '{'.'.join(k)}' due to shape mismatch: "
                      f"{source_val.shape} vs {v.shape}")
                missing_keys.append('.'.join(k))
        else:
            missing_keys.append('.'.join(k))

    params = freeze(unflatten_dict(flat_params))

    # --- Handle batch_stats ---
    flat_state = flatten_dict(var_unfrozen['batch_stats'])
    flat_source_state = {
        tuple(k.split('.')): v for k, v in source_state.items()
    }

    for k, v in flat_state.items():
        if k in flat_source_state:
            source_val = flat_source_state[k]
            if v.shape == source_val.shape:
                flat_state[k] = source_val
            else:
                print(f"Skipping loading batch stat '{'.'.join(k)}' due to shape mismatch: "
                      f"{source_val.shape} vs {v.shape}")
                missing_keys.append('.'.join(k))
        else:
            missing_keys.append('.'.join(k))

    batch_stats = freeze(unflatten_dict(flat_state))

    # --- Log ---
    source_param_keys = set(flat_source_params.keys())
    loaded_param_keys = set(flat_params.keys())
    unexpected_keys.extend(['.'.join(k) for k in source_param_keys - loaded_param_keys])

    source_state_keys = set(flat_source_state.keys())
    loaded_state_keys = set(flat_state.keys())
    unexpected_keys.extend(['.'.join(k) for k in source_state_keys - loaded_state_keys])

    if missing_keys:
        print(f'⚠️  WARNING: {len(missing_keys)} missing keys while loading state_dict.\n→ {missing_keys}')
    if unexpected_keys:
        print(f'⚠️  WARNING: {len(unexpected_keys)} unexpected keys found in pretrained state_dict.\n→ {unexpected_keys}')

    return dict(params=params, batch_stats=batch_stats)


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

from config.optimizer import get_velo_optimizer

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

from flax.core import freeze, unfreeze

from flax.core import freeze
import jax
import jax.numpy as jnp
import optax
import time
from tqdm import trange, tqdm
import functools

def create_model(variant, pretrained=False, rng=None, input_shape=None, dtype=jnp.float32, **kwargs):
    model_cfg = get_model_cfg(variant)
    model_args = model_cfg['arch_fn'](variant, **model_cfg['arch_cfg'])
    model_args.update(kwargs)


    model_args['num_classes'] = kwargs.get('num_classes', model_args.get('num_classes', 1000))  # NEW


    # resolve some special layers and their arguments
    se_args = model_args.pop('se_cfg', {})  # not consumable by model
    if 'se_layer' not in model_args:
        if 'bound_act_fn' in se_args:
            se_args['bound_act_fn'] = get_act_fn(se_args['bound_act_fn'])
        if 'gate_fn' in se_args:
            se_args['gate_fn'] = get_act_fn(se_args['gate_fn'])
        model_args['se_layer'] = partial(SqueezeExcite, **se_args)

    bn_args = model_args.pop('bn_cfg')  # not consumable by model
    if 'norm_layer' not in model_args:
        model_args['norm_layer'] = partial(batchnorm2d, **bn_args)

    model_args['act_fn'] = get_act_fn(model_args.pop('act_fn', 'relu'))  # convert str -> fn

    model = EfficientNet(dtype=dtype, default_cfg=model_cfg['default_cfg'], **model_args)
    
    model.default_cfg['num_classes'] = model_args['num_classes']  # NEW


    rng = jax.random.PRNGKey(0) if rng is None else rng
    params_rng, dropout_rng = jax.random.split(rng)
    input_shape = model_cfg['default_cfg']['input_size'] if input_shape is None else input_shape
    #input_shape = (1, input_shape[1], input_shape[2], input_shape[0])   # CHW -> HWC by default
    input_shape = (1,) + input_shape  # NHWC

    # FIXME is jiting the init worthwhile for my usage?
    #     @jax.jit
    #     def init(*args):
    #         return model.init(*args, training=True)

    variables = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        jnp.ones(input_shape, dtype=dtype),
        training=False)
    
    if pretrained:
        variables = load_pretrained(variables, default_cfg=model.default_cfg, filter_fn=_filter)


    return model, variables

# ---- Helper: create a mask for Optax ----
def mask_head_only(params):
    """Create a PyTree mask where only the 'head' parameters are True (trainable)."""
    return jax.tree_util.tree_map_with_path(lambda path, _: path[0] == 'head', params)

# ---- Main train function ----
def train(train_ds, test_ds, ds_info, model, hparams: dict, 
          rng=None, opt='adam', nb_epochs=10, train_batch_size=32, test_batch_size=32):
    
    print(f"################### TRAINING WITH {opt} ###################")

    num_classes = ds_info.features["label"].num_classes
    model = model.lower()
    opt = opt.lower()
    
    net, variables = create_model(
        variant="tf_efficientnet_l2_ns",
        pretrained=True,
        input_shape=(224, 224, 3),  # adjust as needed
        num_classes=10
    )
    init_params = variables['params']
    init_batch_stats = variables.get('batch_stats', {})

    def predict(params, inputs, aux, train=False):
        all_params = {"params": params, "batch_stats": aux}
        return net.apply(all_params, inputs, training=train, mutable=["batch_stats"] if train else False)
            
    @jax.jit
    def compute_softmax_loss(params, logits, labels):
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        
    @jax.jit
    def accuracy_and_loss(params, data, aux):
        inputs, labels = data
        logits = predict(params, inputs, aux, train=False)
        pred_class = jnp.argmax(logits, axis=-1)
        true_class = jnp.argmax(labels, axis=-1)
        acc = jnp.mean(pred_class == true_class)
        loss_val = compute_softmax_loss(params, logits, labels)
        return acc, loss_val

    def loss_fun(params, data, aux):
        inputs, labels = data
        logits, net_state = predict(params, inputs, aux, train=True)
        loss_val = compute_softmax_loss(params, logits, labels)
        return loss_val, net_state["batch_stats"]

    iter_per_epoch_train = ds_info.splits["train[:90%]"].num_examples // train_batch_size
    NUM_STEPS = nb_epochs * iter_per_epoch_train

    LR = hparams['lr']
    MOMENTUM = hparams['momentum']

    # ---- OPTIMIZER MASKING ----
    mask = mask_head_only(init_params)
    base_opt = optax.masked(optax.adam(LR), mask)

    solver = OptaxSolver(
        opt=base_opt,
        fun=jax.value_and_grad(loss_fun, has_aux=True),
        maxiter=NUM_STEPS,
        has_aux=True,
        value_and_grad=True,
    )

    params = init_params
    batch_stats = init_batch_stats
    print("Classifier kernel shape:", params['head']['classifier']['kernel'].shape)

    state = solver.init_state(params, next(test_ds), batch_stats)
    jitted_update = jax.jit(solver.update)

    print(f"\nnumber of parameters for {model}: {count_parameters(params)}\n")
    print(f"Training {NUM_STEPS} steps over {nb_epochs} epochs")

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    step_train_acc, step_train_loss = [], []
    epoch_times = []
    total_start = time.time()

    test_data = list(test_ds)

    for epoch in trange(nb_epochs, desc="Epochs"):
        epoch_start = time.time()
        pbar = tqdm(range(iter_per_epoch_train), leave=False, desc="Steps")

        for step in pbar:
            train_minibatch = next(train_ds)
            acc_train, loss_train = accuracy_and_loss(params, train_minibatch, batch_stats)

            # log step accuracy/loss
            step_train_acc.append(float(acc_train))
            step_train_loss.append(float(loss_train))
            
            # update
            params, state = jitted_update(params=params, state=state, data=train_minibatch, aux=batch_stats)
            batch_stats = state.aux
            pbar.set_postfix(acc=acc_train.item(), loss=loss_train.item())

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # evaluate train
        train_eval_batches = [next(train_ds) for _ in range(iter_per_epoch_train)]
        train_metrics = [accuracy_and_loss(params, batch, batch_stats) for batch in train_eval_batches]
        train_acc = float(jnp.mean(jnp.array([m[0] for m in train_metrics])))
        train_loss = float(jnp.mean(jnp.array([m[1] for m in train_metrics])))

        # evaluate test
        test_metrics = [accuracy_and_loss(params, batch, batch_stats) for batch in test_data]
        test_acc = float(jnp.mean(jnp.array([m[0] for m in test_metrics])))
        test_loss = float(jnp.mean(jnp.array([m[1] for m in test_metrics])))

        tqdm.write(f"[Epoch {epoch+1}] Train acc: {train_acc:.3f}, loss: {train_loss:.3f} | "
                   f"Test acc: {test_acc:.3f}, loss: {test_loss:.3f} | Time: {epoch_time:.2f}s")

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    total_time = time.time() - total_start
    print(f"Total training time: {total_time:.2f}s")

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




results = train(
    train_ds=iter(train_loader),
    test_ds=iter(val_loader),
    ds_info=ds_info,
    model="efficientnet",
    hparams={"lr": 1e-4, "momentum": 0.9},
    rng=jax.random.PRNGKey(0),
    opt="adam",
    nb_epochs=2,
    train_batch_size=32,
    test_batch_size=32,
)


