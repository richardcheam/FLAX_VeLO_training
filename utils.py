"""
utils.py

A collection of utility/helper functions for data loading, result serialization, 
and visualization of training metrics for deep learning experiments.

Includes:
- TensorFlow Datasets loading with preprocessing.
- Pickle-based saving/loading for metrics and results.
- Accuracy, loss, and training time visualization with Matplotlib.

Designed for research experiments comparing optimizers (e.g., VeLO vs. Adam)
across datasets like MNIST, KMNIST, FashionMNIST, and CIFAR-100 using models like ResNet.

Author: Richard CHEAM
Date: 26th May 2025
"""

# Dataset loading
import tensorflow_datasets as tfds
import tensorflow as tf

# File serialization
import pickle
from flax import serialization

# Plotting
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os

import argparse
from models.resnet import ResNet1, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


#################################################### PLOT ###################################################### 

def save_plot(
    fig,
    model: str,
    dataset: str,
    plot_type: str,
    folder: str = "results/figures",
    extension: str = "png",
    dpi: int = 80
):
    """
    Save the plot to a file under results/figures/<dataset>/.

    Args:
        fig: matplotlib Figure object
        model: model name (used in filename)
        dataset: dataset name (used in path and filename)
        plot_type: e.g., "test_acc", "step_train_loss"
        folder: root folder for saving
        extension: file type to save (e.g., "png", "pdf")
        dpi: resolution for saved file
    """
    os.makedirs(os.path.join(folder, dataset), exist_ok=True)
    filename = os.path.join(folder, dataset, f"{model}_{plot_type}.{extension}")
    fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {filename}")


# def plot_step_metric(metric1, metric2, label1, label2, ylabel, title, steps_per_epoch, label_every):
#     def add_epoch_ticks(ax, steps_per_epoch, total_steps, label_every):
#         epoch_positions = [i * steps_per_epoch for i in range(1, total_steps // steps_per_epoch + 1)]
#         epoch_labels = [str(i) if i % label_every == 0 else '' for i in range(1, len(epoch_positions)+1)]
#         visible_ticks = [pos for pos, lbl in zip(epoch_positions, epoch_labels) if lbl != '']
#         visible_labels = [lbl for lbl in epoch_labels if lbl != '']
#         ax2 = ax.twiny()
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xticks(visible_ticks)
#         ax2.set_xticklabels(visible_labels, color="red")
#         ax2.set_xlabel("Epoch", color="red")
#         return ax2, visible_ticks

#     fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
#     ax.plot(metric1, label=label1, alpha=0.8)
#     ax.plot(metric2, label=label2, alpha=0.8)

#     total_steps = len(metric1)
#     ax2, visible_ticks = add_epoch_ticks(ax, steps_per_epoch, total_steps, label_every)
#     for tick in visible_ticks:
#         ax.axvline(x=tick, color="red", linestyle="--", alpha=1)

#     ax.set_xlabel("Training Step")
#     ax.set_ylabel(ylabel)
#     ax.set_title(title, fontweight="heavy")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     return fig

# def plot_epoch_accuracy(velo_metrics, alternate_metrics, dataset, model_name, opt1_name, opt2_name, tick_step=5):
#     """
#     Returns a matplotlib Figure for training and test accuracy over epochs.

#     Args:
#         velo_metrics (dict): Accuracy metrics for the VeLO optimizer.
#         alternate_metrics (dict): Accuracy metrics for the alternate optimizer.
#         dataset (str): Dataset name (e.g., 'mnist').
#         model_name (str): Model name (default: 'ResNet18').
#         tick_step (int): Tick interval for x-axis.
    
#     Returns:
#         matplotlib.figure.Figure
#     """
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
#     range_epochs = list(range(0, len(velo_metrics['train_acc'])))

#     ax.plot(range_epochs, velo_metrics['train_acc'], label=f"{opt1_name} Train", color="crimson", linestyle='--')
#     ax.plot(range_epochs, alternate_metrics['train_acc'], label=f"{opt2_name} Train", color="steelblue", linestyle='--')
#     ax.plot(range_epochs, velo_metrics['test_acc'], label=f"{opt1_name} Test", color="crimson", linestyle='-')
#     ax.plot(range_epochs, alternate_metrics['test_acc'], label=f"{opt2_name} Test", color="steelblue", linestyle='-')

#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Accuracy")
#     ax.set_title(f"Accuracy Curve of {model_name} on {dataset}", fontweight="bold")
#     ax.set_xticks(range(0, len(range_epochs) + 1, tick_step))
#     ax.legend(loc='lower right')
#     plt.tight_layout()
#     plt.show()
#     return fig

# def plot_epoch_loss(velo_metrics, alternate_metrics, dataset, model_name, opt1_name, opt2_name, tick_step=5):
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
#     range_epochs = list(range(0, len(velo_metrics['train_loss'])))

#     ax.plot(range_epochs, velo_metrics['train_loss'], label=f"{opt1_name} Train", color="crimson", linestyle='--')
#     ax.plot(range_epochs, alternate_metrics['train_loss'], label=f"{opt2_name} Train", color="steelblue", linestyle='--')
#     ax.plot(range_epochs, velo_metrics['test_loss'], label=f"{opt1_name} Test", color="crimson", linestyle='-')
#     ax.plot(range_epochs, alternate_metrics['test_loss'], label=f"{opt2_name} Test", color="steelblue", linestyle='-')

#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Loss")
#     ax.set_title(f"Loss Curve of {model_name} on {dataset}", fontweight="bold")
#     ax.set_xticks(range(0, len(range_epochs) + 1, tick_step))
#     ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()
#     return fig

# def plot_epoch_time_curve(velo_metrics, alternate_metrics, dataset, model_name, opt1_name, opt2_name, tick_step=5):
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
#     range_epochs = list(range(1, len(velo_metrics['epoch_times']) + 1))

#     ax.plot(range_epochs, velo_metrics['epoch_times'], label=f"{opt1_name} Train", marker='o', linestyle='-', color='crimson')
#     ax.plot(range_epochs, alternate_metrics['epoch_times'], label=f"{opt2_name} Train", marker='o', linestyle='-', color='steelblue')

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Time per Epoch (seconds)")
#     ax.set_title(f"Epoch Timing Curve for {model_name} on {dataset}", fontweight="bold")
#     ax.set_xticks(range(0, len(range_epochs) + 1, tick_step))
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     return fig

# def plot_total_training_times_bar(results_dict, opt1, opt2, model_name="ResNet18"):
#     datasets = list(results_dict.keys())
    
#     velo_times = [results_dict[ds][opt1] for ds in datasets]
#     adam_times = [results_dict[ds][opt2] for ds in datasets]

#     x = range(len(datasets))
#     width = 0.2

#     fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
#     ax.bar([i + width/2 for i in x], velo_times, width, label=opt1, color="crimson")
#     ax.bar([i - width/2 for i in x], adam_times, width, label=opt2, color="steelblue")

#     ax.set_xticks(x)
#     ax.set_xticklabels(datasets)
#     ax.set_ylabel("Total Training Time (seconds)")
#     ax.set_title(f"Total Training Time Comparison ({model_name})", fontweight="bold")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     return fig

#####################################################################################################################

def save_pickle(data, filename="times.pkl"):
    """
    Saves a Python object to disk using pickle serialization.

    Args:
        data (Any): The Python object to serialize (e.g., dict, list, NumPy array).
        filename (str): The file path where the object will be saved (default: "times.pkl").
    
    Notes:
        - Overwrites any existing file at the specified path.
        - Useful for saving metrics, results, or experiment metadata.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filename="times.pkl"):
    """
    Loads a Python object from a pickle file.

    Args:
        filename (str): The file path to the pickle file (default: "times.pkl").

    Returns:
        Any: The deserialized Python object.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file is not a valid pickle format.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
    
#####################################################################################################################
def save_checkpoint(params, batch_stats, path):
    to_save = {"params": params, "batch_stats": batch_stats}
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(to_save))
    print(f"Checkpoint is written to {path}")

def build_net(model_name: str, num_classes: int):
    model_name = model_name.lower()
    net_cls = {
        "resnet1": ResNet1,
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50
    }.get(model_name)
    if net_cls is None:
        raise ValueError(f"Unknown model type '{model_name}'.")
    return net_cls(num_classes=num_classes)

from evaluate import EvalState
import optax
def restore_checkpoint(ckpt_path, net):

    with open(ckpt_path, "rb") as f:
        raw_bytes = f.read()

    data = serialization.msgpack_restore(raw_bytes)  # {'params', 'batch_stats', 'nb_steps'}

    params       = data["params"]
    batch_stats  = data["batch_stats"]
    #l2reg = data["l2reg"]
    
    state = EvalState.create(
        apply_fn    = net.apply, #forward pass of model 
        params      = params, 
        tx          = optax.sgd(0.01), # we still need an optax transform to satisfy TrainState, but it will never be used
        batch_stats = batch_stats,
        #l2reg       = l2reg
    )
    return state
###################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from glob import glob

#return dictionary of length num_seeds
def load_seeded_metrics(metric_dir: str, model: str, dataset: str, opt: str) -> Dict[int, Dict]:
    """
    Load saved metrics from multiple seeds for a given model, dataset, and optimizer.
    Returns a dict mapping seed number to its metrics dict.
    """
    seed_metrics = {}
    files = glob(os.path.join(metric_dir, f"{model}_{dataset}_{opt}_seed*.pkl"))
    for f in sorted(files):
        seed = int(f.split("_seed")[-1].split(".pkl")[0])
        with open(f, "rb") as fp:
            seed_metrics[seed] = pickle.load(fp)
    return seed_metrics

#mean,std of a metric of length epochs or steps
def compute_mean_std(metric_dict: Dict[int, Dict], metric_name: str):
    """
    Compute mean and std across seeds for a given metric (e.g., test_acc).
    Returns (mean, std) arrays of shape (num_epochs or num_steps,)
    """
    arrays = [np.array(metrics[metric_name]) for metrics in metric_dict.values()]
    stacked = np.stack(arrays)  # shape: (num_seeds, num_points)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return mean, std

#step-level without EMA
def plot_step_metric_all_optimizers(
    optimizer_metrics: Dict[str, Dict[int, Dict]],
    metric_name: str = "step_train_acc",
    ylabel: str = "Step Accuracy",
    title: str = "Training Accuracy over Steps",
    colors: Dict[str, str] = None
):
    """
    Plot step-level training metric (mean ± std) for multiple optimizers.
    Returns the matplotlib figure object for saving.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = {opt_name: default_colors[i % len(default_colors)]
                  for i, opt_name in enumerate(optimizer_metrics)}

    for opt_name, seed_dict in optimizer_metrics.items():
        def get_mean_std(metric_name):
            arrays = [np.array(metrics[metric_name]) for metrics in seed_dict.values()]
            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]
            stacked = np.stack(arrays)
            return np.mean(stacked, axis=0), np.std(stacked, axis=0)

        color = colors[opt_name]
        mean, std = get_mean_std(metric_name)
        steps = np.arange(len(mean))

        ax.plot(steps, mean, label=opt_name, color=color)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


#step-level with EMA and shift view
def exponential_moving_average(arr: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Apply EMA smoothing to a 1D array."""
    smoothed = np.zeros_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * arr[i]
    return smoothed

def plot_smoothed_step_metric(
    optimizer_metrics: Dict[str, Dict[int, Dict]],
    metric_name: str = "step_train_acc",
    ylabel: str = "Step Accuracy",
    title: str = "Training Accuracy per Step (Smoothed)",
    alpha: float = 0.9,
    alpha_fill: float = 0.2,
    initial_shift: int = 500,
    colors: Dict[str, str] = None,
    legend: bool = True
):
    """
    Plot smoothed step-level metric with zoomed-in y-axis and confidence interval.
    Returns the matplotlib figure object for saving.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = {opt_name: default_colors[i % len(default_colors)]
                  for i, opt_name in enumerate(optimizer_metrics)}

    all_min_curves = []
    all_max_curves = []

    for opt_name, seed_dict in optimizer_metrics.items():
        color = colors[opt_name]
        arrays = [np.array(metrics[metric_name]) for metrics in seed_dict.values()]
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]

        stacked = np.stack(arrays)
        mean = np.mean(stacked, axis=0)
        min_c = np.min(stacked, axis=0)
        max_c = np.max(stacked, axis=0)

        # Smooth
        mean_smooth = exponential_moving_average(mean, alpha)
        min_smooth = exponential_moving_average(min_c, alpha)
        max_smooth = exponential_moving_average(max_c, alpha)

        x = np.arange(len(mean_smooth))

        ax.plot(x, mean_smooth, label=opt_name, color=color, linewidth=2)
        ax.fill_between(x, min_smooth, max_smooth, alpha=alpha_fill, color=color)

        all_min_curves.append(min_smooth)
        all_max_curves.append(max_smooth)

    # Adjust y-limits to zoom in (from initial_shift onward)
    if all_min_curves and all_max_curves and initial_shift < len(all_min_curves[0]):
        lower = min(np.min(c[initial_shift:]) for c in all_min_curves)
        upper = max(np.max(c[initial_shift:]) for c in all_max_curves)
        ax.set_ylim(lower, upper)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
    fig.tight_layout()

    return fig


#train_test acc/loss plot
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def plot_train_and_test_metrics(
    optimizer_metrics: Dict[str, Dict[int, Dict]],
    train_metric: str = "train_acc",
    test_metric: str = "test_acc",
    ylabel: str = "Accuracy",
    title: str = "Train vs Test Accuracy",
    colors: Dict[str, str] = None
):
    """
    Plot train and test metrics (mean ± std) for multiple optimizers,
    using the same color for both, but different line styles.
    Returns the matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = {opt_name: default_colors[i % len(default_colors)]
                  for i, opt_name in enumerate(optimizer_metrics)}

    for opt_name, seed_dict in optimizer_metrics.items():
        def get_mean_std(metric_name):
            arrays = [np.array(metrics[metric_name]) for metrics in seed_dict.values()]
            min_len = min(len(arr) for arr in arrays)
            arrays = [arr[:min_len] for arr in arrays]
            stacked = np.stack(arrays)
            return np.mean(stacked, axis=0), np.std(stacked, axis=0)

        color = colors[opt_name]
        train_mean, train_std = get_mean_std(train_metric)
        test_mean, test_std = get_mean_std(test_metric)
        epochs = np.arange(len(train_mean))

        # Plot train (solid line)
        ax.plot(epochs, train_mean, linestyle='-', label=f"{opt_name} (train)", color=color)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color=color)

        # Plot test (dashed line)
        ax.plot(epochs, test_mean, linestyle='--', label=f"{opt_name} (validation)", color=color)
        ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig

#############################################################################################################################