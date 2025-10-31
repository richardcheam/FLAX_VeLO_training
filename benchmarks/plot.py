import sys
sys.path.append('..')
from utils import *



def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    return p.parse_args()

# Editable constant 
args = parse_args()
DATASET = args.dataset
MODEL = args.model

velo_mnist = load_pickle(f"results/metrics/{MODEL.lower()}_{DATASET}_velo.pkl")
adam_mnist = load_pickle(f"results/metrics/{MODEL.lower()}_{DATASET}_adam.pkl")
adamw_mnist = load_pickle(f"results/metrics/{MODEL.lower()}_{DATASET}_adamw.pkl")
sgd_mnist = load_pickle(f"results/metrics/{MODEL.lower()}_{DATASET}_sgd.pkl")
sgd_momentum_mnist = load_pickle(f"results/metrics/{MODEL.lower()}_{DATASET}_sgd_momentum.pkl")

nb_steps_per_epoch = velo_mnist['nb_steps'] // len(velo_mnist['train_acc']) #total_steps // num_epochs

if velo_mnist["step_train_acc"][0] == adam_mnist["step_train_acc"][0] == adamw_mnist["step_train_acc"][0] == sgd_mnist["step_train_acc"][0] == sgd_momentum_mnist["step_train_acc"][0]:
    raise ValueError("Starting point step-wise of optimizer is not same!")

# Step-wise Training Accuracy
#step_acc = plot_step_metric(velo_mnist["step_train_acc"],
#                            adam_mnist["step_train_acc"],
#                            label1="VeLO", label2=f"{ALTERNATE_OPT}",
#                            ylabel="Accuracy", title=f"{DATASET}_{MODEL}",
#                            steps_per_epoch=nb_steps_per_epoch,
#                            label_every=5)
#save_plot(step_acc, MODEL, DATASET, "step_acc")


#step_loss = plot_step_metric(velo_mnist["step_train_loss"],
#                            adam_mnist["step_train_loss"],
#                            label1="VeLO", label2=f"{ALTERNATE_OPT}",
#                            ylabel="Loss", title=f"{DATASET}_{MODEL}",
#                            steps_per_epoch=nb_steps_per_epoch,
#                            label_every=5)
#save_plot(step_loss, MODEL, DATASET, "step_loss")

# Epoch-wise Training Accuracy
#epoch_acc = plot_epoch_accuracy(velo_metrics=velo_mnist,
#                                alternate_metrics=adam_mnist,
#                                dataset=DATASET,
#                                model_name=MODEL,
#                                opt1_name=MAIN_OPT,
#                                opt2_name=ALTERNATE_OPT)
#save_plot(epoch_acc, MODEL, DATASET, "epoch_acc")

#epoch_loss = plot_epoch_loss(velo_metrics=velo_mnist,
#                            alternate_metrics=adam_mnist,
#                            dataset=DATASET,
#                            model_name=MODEL,
#                            opt1_name=MAIN_OPT,
#                            opt2_name=ALTERNATE_OPT)
#save_plot(epoch_loss, MODEL, DATASET, "epoch_loss")

# Time taken each epoch
epoch_time = plot_epoch_time_curve(velo_metrics=velo_mnist,
                                alternate_metrics=adam_mnist,
                                dataset=DATASET,
                                model_name=MODEL,
                                opt1_name=MAIN_OPT,
                                opt2_name=ALTERNATE_OPT)
save_plot(epoch_time, MODEL, DATASET, "epoch_time")
##########
