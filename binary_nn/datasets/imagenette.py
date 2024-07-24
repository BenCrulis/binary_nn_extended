from pathlib import Path

import torchvision
from torchvision import transforms

from binary_nn.datasets.commons import get_train_transforms, get_test_transforms


def load_imagenette(config, augment=False):
    imagenette_path = Path(config["imagenette"]["path"]).absolute()
    train_transforms = get_train_transforms(augment)
    test_transforms = get_test_transforms()

    ds = torchvision.datasets.ImageFolder(str(imagenette_path / "train"),
                                          transform=train_transforms)
    test_ds = torchvision.datasets.ImageFolder(str(imagenette_path / "val"),
                                               transform=test_transforms)
    n_classes = 10
    return n_classes, ds, test_ds
