import torch.utils.data
from torchvision.transforms import Normalize, Compose, ToTensor, RandomHorizontalFlip


normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

default_train_transform = Compose(
    [ToTensor(), RandomHorizontalFlip(), normalize]
)

default_test_transform = Compose(
    [ToTensor(), normalize]
)
