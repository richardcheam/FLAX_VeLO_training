import sys
sys.path.append('../')

from evaluate import evaluate
from utils import *
import argparse
from dataloader import TFDSDataLoader


def get_seeded_ckpt(ckpt_dir, model, dataset, opt):
    seed_ckpts = glob(os.path.join(ckpt_dir, f"{dataset}/{opt}_{model}_{dataset}_seed*_pretrained.msgpack"))
    return sorted(seed_ckpts)

def aggregate_metric(metrics):
    mean = np.mean(metrics)
    std = np.std(metrics)
    return mean, std

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--opt", type=str, help="optimizer name")
    return p.parse_args()

args = parse_args()
DATASET = args.dataset
MODEL = args.model
BATCH_SIZE = args.batch_size
OPT = args.opt

# load test data
test_loader = TFDSDataLoader(dataset=DATASET, split="test", is_training=False, batch_size=BATCH_SIZE)
ds_info = test_loader.get_info()
num_classes = ds_info.features["label"].num_classes
iter_per_epoch_test = ds_info.splits["test"].num_examples // BATCH_SIZE
net = build_net(MODEL, num_classes=num_classes)

# get all seeds checkpoints of one optimzier
all_ckpt = get_seeded_ckpt("results/checkpoints", MODEL, DATASET, OPT)

# evaluate each seed checkpoint 
test_acc_list, test_loss_list = [], []
for CKPT in all_ckpt:
    print(f"Evaluate on test set for checkpoint: {CKPT}")
    state = restore_checkpoint(CKPT, net)
    acc, loss = evaluate(state, iter(test_loader))
    test_acc_list.append(acc)
    test_loss_list.append(loss)

# caluclate mean and std of acc/loss
print(f"Aggregated test accuracy: {aggregate_metric(test_acc_list)}")
print(f"Aggregated test loss: {aggregate_metric(test_loss_list)}")
