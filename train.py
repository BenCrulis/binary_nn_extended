import os
import argparse
import time

import yaml

from pathlib import Path

import numpy as np

from torch import nn
import torch.nn.functional
from torch.utils.data import random_split

import torchmetrics
from torchmetrics import Accuracy
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from torchvision.models.mobilenet import MobileNetV2, MobileNetV3
from torchvision.models.vgg import vgg19, vgg19_bn, vgg16_bn
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from mlp_mixer_pytorch import MLPMixer

from binary_nn.algorithms.biprop import Biprop
from binary_nn.algorithms.dfa import DFA
from binary_nn.algorithms.drtp import DRTP, DRTPFast
from binary_nn.algorithms.hsic import HSIC
from binary_nn.algorithms.local_search import LS
from binary_nn.algorithms.pepita import Pepita
from binary_nn.algorithms.spsa import SPSA
from binary_nn.algorithms.spsa_g import SPSAG
from binary_nn.algorithms.spsa_h import SPSAH
from binary_nn.datasets.stanforddogs import StanfordDogs, load_dogs
from binary_nn.datasets.indoor import load_indoor
from binary_nn.datasets.imagenette import load_imagenette
from binary_nn.evaluating.classification import eval_classification, eval_classification_iterator
from binary_nn.evaluating.metrics.common.ordered_list import ordered_list, wandb_table_layers
from binary_nn.evaluating.metrics.gradnorm import GradNorm
from binary_nn.evaluating.metrics.lossMetric import LossMetric
from binary_nn.evaluating.metrics.meanabsgrad import MeanAbsGrad
from binary_nn.evaluating.metrics.saturation import Saturation
from binary_nn.models.common.binary.binarization import apply_binarization_parametrization, replace_activations_to_sign, \
    clip_weights_for_binary_layers, replace_activations_to_bireal
from binary_nn.models.common.utils import count_parameters
from binary_nn.models.fullyconnected import FullyConnected
from binary_nn.models.tweaks.remove_skips import remove_skips
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
    ap.add_argument("--test-set", action="store_true",
                    help="use the test set of the dataset instead of splitting the train set into train/validation set")
    ap.add_argument("--reconstruction", action="store_true")
    ap.add_argument("--augment", action="store_true", help="use data augmentation")
    ap.add_argument("--device", type=int, default=0, help="gpu device index")
    ap.add_argument("--early-stopping-delta", type=float, default=0.001,
                    help="minimum validation loss delta required to not trigger early stopping")
    ap.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")

    # dataset config
    ap.add_argument("--dogs", action="store_true", help="use Stanford Dogs dataset")
    ap.add_argument("--indoor", action="store_true", help="use Scene Recognition dataset")
    ap.add_argument("--mnist", action="store_true", help="use MNIST dataset")
    ap.add_argument("--fashion-mnist", action="store_true", help="use FashionMNIST dataset")
    ap.add_argument("--cifar10", action="store_true", help="use CIFAR10 dataset")

    # model related options
    ap.add_argument("--model", default="MobileNetV2")
    ap.add_argument("--binary-weights", action="store_true")
    ap.add_argument("--binary-act", action="store_true")
    ap.add_argument("--unsat", action="store_true")
    ap.add_argument("--bireal-act", action="store_true")
    ap.add_argument("--biprop", nargs="?", const=0.2, type=float)
    ap.add_argument("--remove-skips", action="store_true")

    # training algorithm
    ap.add_argument("--method", type=str, default="bp", help="training algorithm, one of {bp, dfa, drtp, hsic, pepita}")
    ap.add_argument("--slow-drtp", action="store_true", help="use the slow variant of DRTP")
    ap.add_argument("--bw-init", type=str, default="normal", help="initialization method for backward matrices")

    # learning
    ap.add_argument("--opt", type=str, default="Adam", help="optimizer (default to Adam")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--bs", type=int, default=10, help="batch size")
    ap.add_argument("--epochs", type=int, default=80)

    # MLPMixer
    ap.add_argument("--mlp-mixer-layers", type=int, default=12, help="number of layers in MLPMixer")
    ap.add_argument("--mlp-mixer-dim", type=int, default=512, help="size of hidden dimension in MLPMixer")

    # Fully Connected model
    ap.add_argument("--fc-batchnorm", action="store_true", help="use batchnorm in the fully connected model")

    # algorithm specific options
    # Local search
    ap.add_argument("--mut-prob", type=float, default=0.1, help="mutation probability")
    ap.add_argument("--ls-accuracy", action="store_true", help="override loss, use accuracy as the fitness function")

    # HSIC
    ap.add_argument("--hsic-gamma", type=float, default=2.0, help="HSIC gamma value")
    ap.add_argument("--hsic-mode", type=str, default="biased", help="HSIC mode (biased or plausible)")

    # PEPITA
    ap.add_argument("--pepita-gain", type=float, default=1.0, help="pepita gain for the random backward matrix scale")

    # SPSA
    ap.add_argument("--spsa-c", type=float, default=1e-3, help="c parameter of SPSA")
    ap.add_argument("--spsa-g-pert", type=str, default="drtp", help="type of perturbation for guided SPSA")

    # checkpoint related
    ap.add_argument("--save", action="store_true", help="save checkpoints")
    ap.add_argument("--autosave", action="store_true", help="imply --save, auto save at the end of every epoch")

    # logging
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--tags", nargs="*", help="wandb tags")

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
    elif model_name == "FullyConnected":
        return FullyConnected(num_classes, color_input=args.cifar10, with_batchnorm=args.fc_batchnorm)
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
        return DFA(model_config["output_layer"], init=args.bw_init)
    elif algo_name == "drtp":
        algo = DRTP if args.slow_drtp else DRTPFast
        return algo(model_config["output_layer"], num_classes, init=args.bw_init)
    elif algo_name == "hsic":
        return HSIC(num_classes, gamma=args.hsic_gamma, mode=args.hsic_mode)
    elif algo_name == "pepita":
        return Pepita(num_classes, model_config["output_layer"], gain=args.pepita_gain)
    elif algo_name == "ls":
        return LS(args.mut_prob, num_classes=num_classes, accuracy_fitness=args.ls_accuracy)
    elif algo_name == "spsa":
        return SPSA(args.spsa_c)
    elif algo_name == "spsah":
        return SPSAH(args.spsa_c)
    elif algo_name == "spsag":
        return SPSAG(num_classes, args.spsa_c, args.spsa_g_pert)
    else:
        raise ValueError(f"unknown algorithm: {algo_name}")


def compute_run_name(args):
    name = f"{args.model}-{args.method}"
    if args.method == "spsa" or args.method == "spsah" and args.opt != "SGD":
        name += f"-{args.opt}"
    if args.binary_weights and args.binary_act:
        name += "-binary"
    elif args.biprop is not None:
        name += "-biprop"
        if args.binary_act:
            name += "-signSat"
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

    with open(Path("ds_config.yaml"), mode="r") as file:
        ds_config = yaml.safe_load(file)

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
    bireal_act = args.bireal_act
    unsat = args.unsat
    early_stopping_delta = args.early_stopping_delta
    seed = np.random.randint(2**31)
    seed_everything(seed)

    metric_saturation_threshold = 1.0

    if args.no_wandb:
        os.environ["WANDB"] = "0"
    if os.environ.get("WANDB", "") == "":
        os.environ["WANDB"] = "1"

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    print("using device:", device)

    if args.mnist:
        num_classes = 10
        ds = MNIST("datasets", train=True,
                   transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), download=True)
        test_ds = MNIST("datasets", train=False,
                        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]), download=True)
    elif args.fashion_mnist:
        num_classes = 10
        ds = FashionMNIST("datasets", train=True,
                          transform=Compose([ToTensor(), Normalize((0.2860,), (0.3530,))]), download=True)
        test_ds = FashionMNIST("datasets", train=False,
                               transform=Compose([ToTensor(), Normalize((0.2860,), (0.3530,))]), download=True)
    elif args.cifar10:
        num_classes = 10
        ds = CIFAR10("datasets", train=True,
                     transform=Compose([
                         Resize(224),
                         ToTensor(),
                         Normalize((0.49139968, 0.48215841, 0.44653091),
                                   (0.24703223, 0.24348513, 0.26158784))]), download=True)
        test_ds = CIFAR10("datasets", train=False,
                          transform=Compose([
                              Resize(224),
                              ToTensor(),
                              Normalize((0.49139968, 0.48215841, 0.44653091),
                                        (0.24703223, 0.24348513, 0.26158784))]),
                          download=True)
    elif args.dogs:
        print("Using Stanford Dogs dataset")
        num_classes, ds, test_ds = load_dogs(ds_config, augment=augment)
    elif args.indoor:
        print("Using MIT Indoor Scene Recognition dataset")
        num_classes, ds, test_ds = load_indoor(ds_config, augment=augment)
    else:
        print('Using ImageNette dataset')
        num_classes, ds, test_ds = load_imagenette(ds_config, augment=augment)
    sample = ds[0][0][None, ...]

    algo = load_algorithm(algo_name, config["model_config"][model_name], num_classes, args)

    if not args.test_set:
        train_ds, eval_ds = random_split(ds, [train_fraction, 1.0-train_fraction])
        eval_set = "validation"
    else:
        train_ds, eval_ds = ds, test_ds
        eval_set = "test"

    print(num_classes)

    if reconstruction:
        model = load_reconstruction_model(model_name)
    else:
        model = load_model(model_name, num_classes, args)

    if args.remove_skips:
        remove_skips(model)

    if args.biprop is not None:
        Biprop(model, args.biprop)

    n_params = count_parameters(model)
    print(f"using model {model_name} with {n_params} parameters")

    model_config = config["model_config"][("ae" + model_name) if reconstruction else model_name]

    if binary_act:
        print("applying Sign activations")
        replace_activations_to_sign(model, unsaturating=unsat)
    elif bireal_act:
        print("applying BiReal activations")
        replace_activations_to_bireal(model)
    if binary_weights:
        apply_binarization_parametrization(model, model_config["prevent_binarization"])

    if reconstruction:
        loss_fn = torch.nn.functional.mse_loss
    else:
        loss_fn = torch.nn.functional.cross_entropy
    opt = args.opt

    run_name = compute_run_name(args)
    if os.environ.get("WANDB", "") == "1":
        logger = WandbLogger(project="binary nn extended", name=run_name, config={
            "args": args,
            "eval is test set": args.test_set,
            "model": model_name,
            "model arch": str(model),
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
            "binary-act": binary_act,
            "saturating binary activation": not unsat,
            "binary weights": binary_weights,
            "binary-weights": binary_weights,
            "mutation-rate": args.mut_prob,
            "removed skip connexions": args.remove_skips,
            "save": save,
            "device": str(device),
            "metric saturation threshold": metric_saturation_threshold,
            "mlp-mixer-dim": args.mlp_mixer_dim,
            "mlp-mixer-layers": args.mlp_mixer_layers,
            **(algo.config() if hasattr(algo, "config") else {}),
        }, tags=args.tags)

    metrics = {
        "loss": LossMetric(loss_fn).to(device),
        "accuracy": Accuracy("multiclass", num_classes=num_classes).to(device),
        # "precision": torchmetrics.Precision("multiclass", num_classes=num_classes).to(device),
        # "recall": torchmetrics.Recall("multiclass", num_classes=num_classes).to(device)
    }

    eval_losses = []

    gradnorm_metric = GradNorm(model, sample).to(device)
    meanabsgrad_metric = MeanAbsGrad(model, sample).to(device)
    saturation_metric = Saturation(model, threshold=metric_saturation_threshold)

    model.to(device)

    i = 0
    for epoch, epoch_iterator in enumerate(train(model, train_ds, opt, loss_fn, reconstruction=reconstruction,
                                                 lr=lr, batch_size=bs, num_epochs=epochs, train_callback=algo,
                                                 device=device, num_workers=args.num_workers, opt_kwargs={"weight_decay": wd})):
        print(f"epoch {epoch}")
        for metric in metrics.values():
            metric.reset()
        time_before = time.time()
        gradnorm_metric.reset()
        # saturation_metric.reset()
        # training loop
        batch_pbar = tqdm(epoch_iterator)
        for iteration, l, x, y, y_pred in batch_pbar:
            i += 1
            batch_pbar.set_description(f"l: {l.item():.5f}")

            train_log = {
                    "epoch": epoch,
                    "train/batch loss": l.detach().cpu().item(),
                    # **ordered_list(grad_norms, "train/layer {i} grad norm"),
                    # **ordered_list(sat, "train/layer {i} saturation"),
                }

            for metric_name, metric in metrics.items():
                train_log[f"batch/{metric_name}"] = metric(y_pred, y).cpu()

            if binary_weights:
                clip_weights_for_binary_layers(model)

            if iteration == 0:
                grad_norms = gradnorm_metric()
                mean_abs_grads = meanabsgrad_metric()
                sat = saturation_metric()
                saturation_metric.reset_buffers()
                train_log.update({
                    "train/grad norms": wandb_table_layers(mean_abs_grads, "mean"),
                    "train/mean abs grad": mean_abs_grads,
                    "train/saturation": sat,
                    "train/mean gradient": wandb_table_layers(grad_norms, "mean"),
                    "train/average saturation": wandb_table_layers(sat, "mean"),
                })

            if iteration == len(epoch_iterator) - 1:
                time_epoch = time.time() - time_before
                train_log["time epoch"] = time_epoch

            # print(f"epoch {epoch}, iteration {i}: loss = {l.item()}")
            if os.environ.get("WANDB", "") == "1":
                logger.log(train_log, step=i, commit=False)
        print(f"end of epoch {epoch}")

        eval_log = {}
        for metric_name, metric in metrics.items():
            eval_log[f"train/{metric_name}"] = metric.compute().cpu()

        print("evaluating on validation set")
        for metric in metrics.values():
            metric.reset()
        # saturation_metric.reset()
        pbar = tqdm(eval_classification_iterator(model, eval_ds, metrics, batch_size=bs, device=device, num_workers=args.num_workers))
        for r in pbar:
            pbar.set_description(f"{r}")

        eval_loss = metrics["loss"].compute().cpu().item()
        eval_losses.append(eval_loss)

        for metric_name, metric in metrics.items():
            eval_log[f"{eval_set}/{metric_name}"] = metric.compute().cpu()
        # eval_log[f"{eval_set}/saturation"] = saturation_metric.compute()
        if os.environ.get("WANDB", "") == "1":
            logger.log(eval_log, step=i, commit=True)

        print("end of eval")

        if len(eval_losses) >= 5 and (eval_losses[-5] - eval_losses[-1]) <= early_stopping_delta:
            print("No improvement over the last 5 epochs, stopping run.")
            return
        pass

    return


if __name__ == '__main__':
    main()
    print("end of program")
