import argparse

import yaml

from pathlib import Path

import numpy as np

from torch import nn
import torch.nn.functional
from torch.utils.data import random_split

import torchmetrics
from torchmetrics import Accuracy

from torchvision.models.mobilenet import MobileNetV2, MobileNetV3
from torchvision.models.vgg import vgg19, vgg19_bn, vgg16_bn
from mlp_mixer_pytorch import MLPMixer

from binary_nn.algorithms.dfa import DFA
from binary_nn.algorithms.drtp import DRTP
from binary_nn.algorithms.local_search import LS
from binary_nn.algorithms.spsa import SPSA
from binary_nn.datasets.imagenette import load_imagenette
from binary_nn.evaluating.classification import eval_classification, eval_classification_iterator
from binary_nn.evaluating.metrics.lossMetric import LossMetric
from binary_nn.models.common.binary.binarization import apply_binarization_parametrization, replace_activations_to_sign, \
    clip_weights_for_binary_layers
from binary_nn.models.common.utils import count_parameters
from binary_nn.training.training import train
from binary_nn.models import autoencoders as ae

from tqdm import tqdm

from utils.logging.wandblogger import WandbLogger
from utils.seed import seed_everything


def parse_args():
    ap = argparse.ArgumentParser("Binary NNs")
    # general options
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train-fraction", type=float, default=0.9)
    ap.add_argument("--reconstruction", action="store_true")
    ap.add_argument("--augment", action="store_true", help="use data augmentation")

    # model related options
    ap.add_argument("--model", default="MobileNetV2")
    ap.add_argument("--binary-weights", action="store_true")
    ap.add_argument("--binary-act", action="store_true")
    ap.add_argument("--unsat", action="store_true")

    # training algorithm
    ap.add_argument("--method", type=str, default="bp", help="training algorithm, one of {bp, dfa, drtp}")

    # learning
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--bs", type=int, default=10, help="batch size")
    ap.add_argument("--epochs", type=int, default=80)

    # MLPMixer
    ap.add_argument("--mlp-mixer-layers", type=int, default=12, help="number of layers in MLPMixer")
    ap.add_argument("--mlp-mixer-dim", type=int, default=512, help="size of hidden dimension in MLPMixer")

    # algorithm specific options
    # Local search
    ap.add_argument("--mut-prob", type=float, default=0.1, help="mutation probability")
    ap.add_argument("--ls-accuracy", action="store_true", help="override loss, use accuracy as the fitness function")

    # SPSA
    ap.add_argument("--spsa-c", type=float, default=1e-3, help="c parameter of SPSA")

    # checkpoint related
    ap.add_argument("--save", action="store_true", help="save checkpoints")
    ap.add_argument("--autosave", action="store_true", help="imply --save, auto save at the end of every epoch")

    # logging
    ap.add_argument("--no-wandb", action="store_true")

    return ap.parse_args()


def load_model(model_name, num_classes, args):
    if model_name == "MobileNetV2":
        return MobileNetV2(num_classes=num_classes)
    elif model_name == "vgg16":
        model = vgg16_bn()
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model
    elif model_name == "vgg19":
        model = vgg19_bn()
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model
    elif model_name == "MLPMixer":
        return MLPMixer(image_size=224, channels=3, patch_size=16, dim=args.mlp_mixer_dim,
                        depth=args.mlp_mixer_layers, num_classes=num_classes)
    else:
        raise ValueError(f"unknown model: {model_name}")


def load_reconstruction_model(model_name):
    if model_name == "vgg11":
        model = ae.vgg.VGGAutoEncoder(ae.vgg.get_configs("vgg11"))
    elif model_name == "vgg13":
        model = ae.vgg.VGGAutoEncoder(ae.vgg.get_configs("vgg13"))
    elif model_name == "vgg16":
        model = ae.vgg.VGGAutoEncoder(ae.vgg.get_configs("vgg16"))
    elif model_name == "vgg19":
        model = ae.vgg.VGGAutoEncoder(ae.vgg.get_configs("vgg19"))
    else:
        raise ValueError(f"unknown model: {model_name}")
    return model


def load_algorithm(algo_name, model_config, num_classes, args):
    if algo_name == "bp":
        return None
    elif algo_name == "dfa":
        return DFA(model_config["output_layer"])
    elif algo_name == "drtp":
        return DRTP(model_config["output_layer"], num_classes)
    elif algo_name == "ls":
        return LS(args.mut_prob, num_classes=num_classes, accuracy_fitness=args.ls_accuracy)
    elif algo_name == "spsa":
        return SPSA(args.spsa_c)
    else:
        raise ValueError(f"unknown algorithm: {algo_name}")


def compute_run_name(args):
    name = f"{args.model}-{args.method}"
    if args.binary_weights and args.binary_act:
        name += "-binary"
    elif args.binary_weights:
        name += "-binary-weights"
    elif args.binary_act:
        name += "-binary-act"
    return name


def main():
    args = parse_args()

    if args.autosave:
        args.save = True

    print(args)

    config_path = args.config
    with open(config_path, mode="r") as file:
        config = yaml.safe_load(file)

    print(config)

    save = args.save
    autosave = args.autosave

    reconstruction = args.reconstruction
    augment = args.augment
    train_fraction = args.train_fraction
    algo_name = args.method
    model_name = args.model
    lr = args.lr
    wd = args.wd
    bs = args.bs
    epochs = args.epochs
    binary_weights = args.binary_weights
    binary_act = args.binary_act
    unsat = args.unsat
    seed = np.random.randint(2**31)
    seed_everything(seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("using device:", device)

    num_classes, ds, test_ds = load_imagenette(config, augment=augment)

    algo = load_algorithm(algo_name, config["model_config"][model_name], num_classes, args)

    train_ds, validation_ds = random_split(ds, [train_fraction, 1.0-train_fraction])

    print(num_classes)

    if reconstruction:
        model = load_reconstruction_model(model_name)
    else:
        model = load_model(model_name, num_classes, args)

    n_params = count_parameters(model)
    print(f"using model {model_name} with {n_params}")

    model_config = config["model_config"][("ae" + model_name) if reconstruction else model_name]

    if binary_act:
        replace_activations_to_sign(model, unsaturating=unsat)
    if binary_weights:
        apply_binarization_parametrization(model, model_config["prevent_binarization"])

    if reconstruction:
        loss_fn = torch.nn.functional.mse_loss
    else:
        loss_fn = torch.nn.functional.cross_entropy
    opt = "Adam"

    run_name = compute_run_name(args)
    logger = WandbLogger(project="binary nn extended", name=run_name, config={
        "args": args,
        "model": model_name,
        "parameters": n_params,
        "optimizer": opt,
        "algorithm": algo_name,
        "lr": lr,
        "wd": wd,
        "bs": bs,
        "total epochs": epochs,
        "seed": seed,
        "reconstruction": reconstruction,
        "augmentation": augment,
        "fraction train": train_fraction,
        "binary activations": binary_act,
        "saturating binary activation": not unsat,
        "binary weights": binary_weights,
        "mutation-rate": args.mut_prob,
        "save": save,
        "device": str(device),
    })

    metrics = {
        "loss": LossMetric(loss_fn).to(device),
        "accuracy": Accuracy("multiclass", num_classes=num_classes).to(device),
        "precision": torchmetrics.Precision("multiclass", num_classes=num_classes).to(device),
        "recall": torchmetrics.Recall("multiclass", num_classes=num_classes).to(device)
    }

    model.to(device)

    i = 0
    for epoch, epoch_iterator in enumerate(train(model, train_ds, opt, loss_fn, reconstruction=reconstruction,
                                                 lr=lr, batch_size=bs, num_epochs=epochs, train_callback=algo,
                                                 device=device, opt_kwargs={"weight_decay": wd})):
        print(f"epoch {epoch}")
        # training loop
        for iteration, l, x, y, y_pred in tqdm(epoch_iterator):
            i += 1

            if binary_weights:
                clip_weights_for_binary_layers(model)

            # print(f"epoch {epoch}, iteration {i}: loss = {l.item()}")
            logger.log({
                "epoch": epoch,
                "train/batch loss": l.detach().cpu().item(),
            }, step=i, commit=False)
        print(f"end of epoch {epoch}")
        print("evaluating on validation set")
        for metric in metrics.values():
            metric.reset()
        pbar = tqdm(eval_classification_iterator(model, validation_ds, metrics, batch_size=bs, device=device))
        for r in pbar:
            pbar.set_description(f"{r}")

        eval_log = {}
        for metric_name, metric in metrics.items():
            eval_log[f"validation/{metric_name}"] = metric.compute().cpu()
        logger.log(eval_log, step=i, commit=True)

        print("end of eval")

    return


if __name__ == '__main__':
    main()
    print("end of program")
