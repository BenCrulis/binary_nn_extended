from pathlib import Path

import torchvision
from torchvision import transforms

from binary_nn.datasets.commons import default_train_transform, default_test_transform


def load_imagenette(config, augment=False):
    imagenette_path = Path(config["datasets"]["imagenette"]["path"]).absolute()
    if augment:
        train_transforms = [transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20),
                            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), ]
    else:
        train_transforms = [transforms.Resize((224, 224))]
    train_transforms.append(default_train_transform)
    if augment:
        train_transforms.append(transforms.RandomErasing(0.9))
    ds = torchvision.datasets.ImageFolder(str(imagenette_path / "train"),
                                          transform=transforms.Compose(train_transforms))
    test_ds = torchvision.datasets.ImageFolder(str(imagenette_path / "val"),
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),
                                                   default_test_transform]))
    n_classes = 10
    return n_classes, ds, test_ds
