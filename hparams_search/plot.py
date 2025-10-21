import argparse
import optuna
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure 

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--use_lopt", action="store_true", help="True if train with VeLO, else use alt_opt")
    p.add_argument("--alt_opt", default="adam", help="adam | sgd")
    return p.parse_args()

def save_plot(fig, model: str, dataset: str, plot_type: str):
    filename = f"{plot_type}_{model}_{dataset}.png"
    fig.savefig(f"figures/{filename}", bbox_inches="tight")
    plt.close(fig)

args = parse_args()
DATASET = args.dataset
MODEL = args.model
USE_LOPT = args.use_lopt
ALTERNATE_OPT = args.alt_opt
OPT = 'velo' if USE_LOPT else ALTERNATE_OPT

db_url = f"sqlite:///study/{OPT}_{MODEL}_{DATASET}.db"
study = optuna.load_study(study_name=f"{OPT}_{MODEL}_{DATASET}", storage=db_url)

#fig = optuna.visualization.plot_intermediate_values(study)
#fig.show()
#save_plot(fig, MODEL, DATASET, "intermediate_val")

fig = optuna.visualization.plot_param_importances(study)
fig.show()
#save_plot(fig, MODEL, DATASET, "hparam_importance")
