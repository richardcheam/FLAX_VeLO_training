import sys
sys.path.append('../')

import optuna
from optuna.pruners import MedianPruner
import jax

from config.hparams import suggest_hparams
from utils import split_train_val 
from config.optimizer import build_optimizer
from trainer import *
from utils import save_checkpoint
import argparse
    
def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--use_lopt", action="store_true", help="True if train with VeLO, else use alt_opt")
    p.add_argument("--alt_opt", default="adam", help="adam | sgd")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_trials", type=int, default=5)
    return p.parse_args()
    
args = parse_args()

DATASET = args.dataset
MODEL = args.model
USE_LOPT = args.use_lopt
ALTERNATE_OPT = args.alt_opt
N_TRIALS = args.n_trials
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
OPT = 'velo' if USE_LOPT else ALTERNATE_OPT
SEED = 42

train_ds, val_ds, ds_info = split_train_val(dataset=DATASET, train_ratio=0.9, batch_size=BATCH_SIZE)

def objective(trial):
    # define search space in config/hparams.py
    HPARAMS = suggest_hparams(trial)

    # use a unique PRNGKey per trial
    KEY = jax.random.PRNGKey(SEED + trial.number)

    results = trainer(
        train_ds, val_ds, ds_info,
        model=MODEL,
        hparams=HPARAMS,
        optimizer_fn=build_optimizer, 
        run_name=f"trial_{trial.number}",
        use_lopt=USE_LOPT,                 
        alternate_opt=OPT,
        nb_epochs=EPOCHS,
        train_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        rng=KEY,
    )
    # choose a metric to optimize (objective), here, maximize final validation acc
    metric = results["test_acc"][-1] #last val acc
    
    filename = f"{OPT}_{MODEL}_{DATASET}_trial_{trial.number}.msgpack"
    ckpt_path = f"checkpoints/{filename}"
    
    save_checkpoint(results["params"],
                    results["batch_stats"],
                    results["nb_steps"],
                    results["l2reg"],
                    path=ckpt_path)	
    
    # store ONLY LIGHT metadata in the study
    trial.set_user_attr("ckpt_path", ckpt_path) #base to find EMA and SWA
                    
    filename = f"ema_{OPT}_{MODEL}_{DATASET}_trial_{trial.number}.msgpack"
    ckpt_path = f"checkpoints/{filename}"    
    save_checkpoint(results["ema_params"],
                    results["batch_stats"],
                    results["nb_steps"],
                    results["l2reg"],
                    path=ckpt_path)	
    
    filename = f"swa_{OPT}_{MODEL}_{DATASET}_trial_{trial.number}.msgpack"
    ckpt_path = f"checkpoints/{filename}"   
    save_checkpoint(results["swa_params"],
                    results["batch_stats"],
                    results["nb_steps"],
                    results["l2reg"],
                    path=ckpt_path)

    return metric  

study = optuna.create_study(
    study_name=f"{OPT}_{MODEL}_{DATASET}",
    direction="maximize",
    storage=f"sqlite:///study/{OPT}_{MODEL}_{DATASET}.db", #store lightweight info from study
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, timeout=None) 


best_trial = study.best_trial
print(f"\nBest hyperparameters: {study.best_params}\n\nsave Best HPs to path: study/{OPT}_{MODEL}_{DATASET}.db")
print("save best checkpoint to:", best_trial.user_attrs["ckpt_path"])



