import torch.utils.data
from torchvision.transforms import (Normalize, Compose, ToTensor, RandomHorizontalFlip, RandomRotation,
                                    RandomResizedCrop, RandomErasing, Resize)


normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

default_train_transform = Compose(
    [ToTensor(), RandomHorizontalFlip(), normalize]
)

default_test_transform = Compose(
    [ToTensor(), normalize]
)


def get_train_transforms(augment):
    if augment:
        train_transforms = [RandomHorizontalFlip(),
                            RandomRotation(20),
                            RandomResizedCrop(224, scale=(0.5, 1.0)), ]
    else:
        train_transforms = [Resize((224, 224))]
    train_transforms.append(default_train_transform)
    if augment:
        train_transforms.append(RandomErasing(0.9))
    return Compose(train_transforms)


def get_test_transforms():
    return Compose([Resize((224, 224)), default_test_transform])
