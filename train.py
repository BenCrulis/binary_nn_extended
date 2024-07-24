import argparse

import yaml

from torch import nn
import torch.nn.functional
from torch.utils.data import random_split

from torchmetrics import Accuracy

import torchvision.datasets
from torchvision.models.mobilenet import MobileNetV2, MobileNetV3
from torchvision.models.vgg import vgg19, vgg19_bn, vgg16_bn

from binary_nn.algorithms.dfa import DFA
from binary_nn.algorithms.drtp import DRTP
from binary_nn.datasets.imagenette import load_imagenette
from binary_nn.evaluating.classification import eval_classification, eval_classification_iterator
from binary_nn.models.common.utils import count_parameters
from binary_nn.training.training import train

from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser("Binary NNs")
    # general options
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train-fraction", type=float, default=0.9)

    # model related options
    ap.add_argument("--model", default="MobileNetV2")
    ap.add_argument("--binary-weights", action="store_true")
    ap.add_argument("--binary-act", action="store_true")

    # training algorithm
    ap.add_argument("--method", type=str, default="bp", help="training algorithm, one of {bp, dfa, drtp}")

    # learning
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)

    # logging
    ap.add_argument("--no-wandb", action="store_true")

    return ap.parse_args()


def load_model(model_name, num_classes):
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
    else:
        raise ValueError(f"unknown model: {model_name}")


def load_algorithm(algo_name, model_config, num_classes):
    if algo_name == "bp":
        return None
    elif algo_name == "dfa":
        return DFA(model_config["output_layer"])
    elif algo_name == "drtp":
        return DRTP(model_config["output_layer"], num_classes)
    else:
        raise ValueError(f"unknown algorithm: {algo_name}")


def main():
    args = parse_args()
    print(args)

    config_path = args.config
    with open(config_path, mode="r") as file:
        config = yaml.safe_load(file)

    print(config)

    train_fraction = args.train_fraction
    algo_name = args.method
    model_name = args.model
    lr = args.lr
    bs = args.bs
    epochs = args.epochs

    num_classes, ds, test_ds = load_imagenette(config, augment=False)

    algo = load_algorithm(algo_name, config["model_config"][model_name], num_classes)

    train_ds, validation_ds = random_split(ds, [train_fraction, 1.0-train_fraction])

    print(num_classes)

    model = load_model(model_name, num_classes)

    n_params = count_parameters(model)
    print(f"using model {model_name} with {n_params}")

    loss_fn = torch.nn.functional.cross_entropy
    opt = "Adam"

    for epoch, epoch_iterator in enumerate(train(model, train_ds, opt, loss_fn,
                                                 lr=lr, batch_size=bs, num_epochs=epochs, train_callback=algo)):
        print(f"epoch {epoch}")
        # training loop
        for i, l, x, y, y_pred in tqdm(epoch_iterator):
            # print(f"epoch {epoch}, iteration {i}: loss = {l.item()}")
            pass
        print(f"end of epoch {epoch}")
        print("evaluating on validation set")
        for r in tqdm(eval_classification_iterator(model, validation_ds,
                                                   {"accuracy": Accuracy("multiclass", num_classes=num_classes)})):
            print(r)

        print("end of eval")

    return


if __name__ == '__main__':
    main()
    print("end of program")
