import sys
sys.path.append('../')

from train import *
from utils import *
from dataloader import *

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    return p.parse_args()

args = parse_args()
DATASET = args.dataset
MODEL = args.model
TRAIN_BATCH_SIZE = args.train_batch_size
TEST_BATCH_SIZE = args.test_batch_size
EPOCHS = args.epochs

print("Using:", jax.devices())

HPARAMS = {
    'lr': 0.01,
    'momentum': 0.9
}

train_split = "train[:90%]" #90%
val_split = "train[90%:]" #10%

for SEED in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

    RNG = jax.random.key(SEED)

    # Load iterators
    train_loader = TFDSDataLoader(dataset=DATASET, split=train_split, is_training=True, batch_size=TRAIN_BATCH_SIZE)
    val_loader = TFDSDataLoader(dataset=DATASET, split=val_split, is_training=False, batch_size=TEST_BATCH_SIZE)
    ds_info = train_loader.get_info()
    # Train with VeLO
    OPT = 'velo'
    velo_metrics = train(iter(train_loader), 
                        iter(val_loader),  
                        ds_info, 
                        model=MODEL,
                        hparams=HPARAMS,
                        rng=RNG,
                        opt=OPT,
                        nb_epochs=EPOCHS,
                        train_batch_size=TRAIN_BATCH_SIZE,
                        test_batch_size=TEST_BATCH_SIZE)
    save_pickle(velo_metrics, f"results/metrics/{MODEL}_{DATASET}_{OPT}_seed{SEED}.pkl")
    filename = f"{OPT}_{MODEL}_{DATASET}_seed{SEED}_pretrained.msgpack"
    save_checkpoint(velo_metrics["params"], velo_metrics["batch_stats"], path=f'results/checkpoints/{ds_info.name}/{filename}') 
    print("###################################################################################################\n")

    # Reload iterators again
    train_loader = TFDSDataLoader(dataset=DATASET, split=train_split, is_training=True, batch_size=TRAIN_BATCH_SIZE)
    val_loader = TFDSDataLoader(dataset=DATASET, split=val_split, is_training=False, batch_size=TEST_BATCH_SIZE)
    OPT = 'sgd'
    # Train with SGD
    sgd_metrics = train(iter(train_loader), 
                        iter(val_loader), 
                        ds_info, 
                        model=MODEL,
                        hparams=HPARAMS,
                        rng=RNG,
                        opt=OPT,
                        nb_epochs=EPOCHS,
                        train_batch_size=TRAIN_BATCH_SIZE,
                        test_batch_size=TEST_BATCH_SIZE)    
    save_pickle(sgd_metrics, f"results/metrics/{MODEL}_{DATASET}_{OPT}_seed{SEED}.pkl")
    filename = f"{OPT}_{MODEL}_{DATASET}_seed{SEED}_pretrained.msgpack"
    save_checkpoint(sgd_metrics["params"], sgd_metrics["batch_stats"], path=f'results/checkpoints/{ds_info.name}/{filename}') 
    print("###################################################################################################\n")

    # Reload iterators again
    train_loader = TFDSDataLoader(dataset=DATASET, split=train_split, is_training=True, batch_size=TRAIN_BATCH_SIZE)
    val_loader = TFDSDataLoader(dataset=DATASET, split=val_split, is_training=False, batch_size=TEST_BATCH_SIZE)
    OPT = 'sgdm'
    # Train with SGD with momentum
    sgdm_metrics = train(iter(train_loader), 
                        iter(val_loader), 
                        ds_info, 
                        model=MODEL,
                        hparams=HPARAMS,
                        rng=RNG,
                        opt=OPT,
                        nb_epochs=EPOCHS,
                        train_batch_size=TRAIN_BATCH_SIZE,
                        test_batch_size=TEST_BATCH_SIZE)
    save_pickle(sgdm_metrics, f"results/metrics/{MODEL}_{DATASET}_{OPT}_seed{SEED}.pkl")
    filename = f"{OPT}_{MODEL}_{DATASET}_seed{SEED}_pretrained.msgpack"
    save_checkpoint(sgdm_metrics["params"], sgdm_metrics["batch_stats"], path=f'results/checkpoints/{ds_info.name}/{filename}') 
    print("###################################################################################################\n")

    # Reload iterators again
    train_loader = TFDSDataLoader(dataset=DATASET, split=train_split, is_training=True, batch_size=TRAIN_BATCH_SIZE)
    val_loader = TFDSDataLoader(dataset=DATASET, split=val_split, is_training=False, batch_size=TEST_BATCH_SIZE)
    OPT = 'adam'
    # Train with Adam
    adam_metrics = train(iter(train_loader), 
                        iter(val_loader), 
                        ds_info, 
                        model=MODEL,
                        hparams=HPARAMS,
                        rng=RNG,
                        opt=OPT,
                        nb_epochs=EPOCHS,
                        train_batch_size=TRAIN_BATCH_SIZE,
                        test_batch_size=TEST_BATCH_SIZE)
    save_pickle(adam_metrics, f"results/metrics/{MODEL}_{DATASET}_{OPT}_seed{SEED}.pkl")
    filename = f"{OPT}_{MODEL}_{DATASET}_seed{SEED}_pretrained.msgpack"
    save_checkpoint(adam_metrics["params"], adam_metrics["batch_stats"], path=f'results/checkpoints/{ds_info.name}/{filename}') 
    print("###################################################################################################\n")

    # Reload iterators again
    train_loader = TFDSDataLoader(dataset=DATASET, split=train_split, is_training=True, batch_size=TRAIN_BATCH_SIZE)
    val_loader = TFDSDataLoader(dataset=DATASET, split=val_split, is_training=False, batch_size=TEST_BATCH_SIZE)
    OPT = 'adamw'
    # Train with Adam weight decay
    adamw_metrics = train(iter(train_loader), 
                        iter(val_loader), 
                        ds_info, 
                        model=MODEL,
                        hparams=HPARAMS,
                        rng=RNG,
                        opt=OPT,
                        nb_epochs=EPOCHS,
                        train_batch_size=TRAIN_BATCH_SIZE,
                        test_batch_size=TEST_BATCH_SIZE)
    save_pickle(adamw_metrics, f"results/metrics/{MODEL}_{DATASET}_{OPT}_seed{SEED}.pkl")
    filename = f"{OPT}_{MODEL}_{DATASET}_seed{SEED}_pretrained.msgpack"
    save_checkpoint(adamw_metrics["params"], adamw_metrics["batch_stats"], path=f'results/checkpoints/{ds_info.name}/{filename}') 

