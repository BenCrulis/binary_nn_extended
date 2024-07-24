import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_iteration(model, x, y, opt: Optimizer, loss_fn):
    y_pred = model(x)
    l = loss_fn(y_pred, y)
    opt.zero_grad()
    l.backward()
    opt.step()

    return l, y_pred


def training_epoch(model, dataloader, opt: Optimizer, loss_fn, device=None):

    for i, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        l, y_pred = train_iteration(model, x, y, opt, loss_fn)

        yield i, l, x, y, y_pred


def train(model: nn.Module,
          dataset,
          optimizer,
          loss_fn,
          lr=1e-3,
          opt_kwargs=None,
          num_epochs=10,
          batch_size=10,
          device=None):
    model = model.to(device)

    if opt_kwargs is None:
        opt_kwargs = {}
    if isinstance(optimizer, str):
        opt_cls = getattr(torch.optim, optimizer)
        opt = opt_cls(model.parameters(), lr=lr, **opt_kwargs)
    elif hasattr(optimizer, "step"):
        opt = optimizer

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for i, l, x, y, y_pred in training_epoch(model, dataloader, opt, loss_fn):
            yield epoch, i, l, x, y, y_pred
