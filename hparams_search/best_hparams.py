import optuna
import argparse

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--use_lopt", action="store_true", help="True if train with VeLO, else use alt_opt")
    p.add_argument("--alt_opt", default="adam", help="adam | sgd")
    return p.parse_args()

args = parse_args()
DATASET = args.dataset
MODEL = args.model
USE_LOPT = args.use_lopt
ALTERNATE_OPT = args.alt_opt
OPT = 'velo' if USE_LOPT else ALTERNATE_OPT

db_url = f"sqlite:///study/{OPT}_{MODEL}_{DATASET}.db"
study = optuna.load_study(study_name=f"{OPT}_{MODEL}_{DATASET}", storage=db_url)
best_trail = study.best_trial

print(f"Best validation accuracy: {best_trail.value:.4f}%")
print("Best Hyperparams:")

for key, value in best_trail.params.items():
    print(f"-> {key}: {value}")
