import sys
sys.path.append('..')
from utils import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, Tuple

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    return p.parse_args()

# Editable constant 
args = parse_args()
MODEL = args.model
DATASET = args.dataset

print(MODEL)
print(DATASET)

optimizer_set = ['VeLO', 'SGD', 'SGDM', 'Adam', 'AdamW']
optimizer_metrics = {}

SHIFT = 5000
ALPHA_SMOOTH = 0.98

for OPT in optimizer_set:
    optimizer_metrics[OPT] = load_seeded_metrics("results/metrics", MODEL, DATASET, OPT.lower())

fig = plot_step_metric_all_optimizers(
    optimizer_metrics=optimizer_metrics,
    metric_name="step_train_acc",
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="step_train_acc_ORIGINAL")

fig = plot_step_metric_all_optimizers(
    optimizer_metrics=optimizer_metrics,
    metric_name="step_train_loss",
    ylabel="Step Loss",
    title="Step Loss over Steps"
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="step_train_loss_ORIGINAL")

fig = plot_smoothed_step_metric(
    optimizer_metrics=optimizer_metrics,
    metric_name="step_train_acc",
    alpha=ALPHA_SMOOTH,
    initial_shift=SHIFT,
    title="Smoothed Step Accuracy (Zoomed)"
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="step_train_acc")

fig = plot_smoothed_step_metric(
    optimizer_metrics=optimizer_metrics,
    metric_name="step_train_loss",
    alpha=ALPHA_SMOOTH,
    initial_shift=SHIFT,
    title="Smoothed Step Loss (Zoomed)"
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="step_train_loss")

#for cifar10 val_acc not test_acc
fig = plot_train_and_test_metrics(
    optimizer_metrics=optimizer_metrics,
    train_metric="train_acc",
    test_metric="val_acc", #test_acc
    ylabel="Accuracy",
    title="Train vs Validation Accuracy",
    zoom_range=(50,75),
    zoom_position="center"
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="train_test_acc")

fig = plot_train_and_test_metrics(
    optimizer_metrics=optimizer_metrics,
    train_metric="train_loss",
    test_metric="val_loss",
    ylabel="Loss",
    zoom_range=(50,75),
    title="Train vs Validation Loss",
    zoom_position="center"
)
save_plot(fig, model=MODEL, dataset=DATASET, plot_type="train_test_loss")

