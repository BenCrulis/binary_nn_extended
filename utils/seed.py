import random
import numpy as np
import torch


def seed_everything(seed: int):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
