import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm


def train_iteration(model, x, y, opt: Optimizer, loss_fn):
    with torch.enable_grad():
        model.train()
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        opt.zero_grad()
        l.backward()
        opt.step()
    return l, y_pred


def training_epoch(model, dataloader, opt: Optimizer, loss_fn, reconstruction=False, train_callback=None, device=None):
    if train_callback is None:
        train_callback = train_iteration

    for i, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        if reconstruction:
            y = x
        else:
            y = y.to(device)

        l, y_pred = train_callback(model, x, y, opt, loss_fn)

        yield i, l, x, y, y_pred


def train(model: nn.Module,
          dataset,
          optimizer,
          loss_fn,
          lr=1e-3,
          opt_kwargs=None,
          num_epochs=10,
          batch_size=10,
          reconstruction=False,
          train_callback=None,
          device=None):
    if train_callback is None:
        train_callback = train_iteration
    model = model.to(device)

    if opt_kwargs is None:
        opt_kwargs = {}
    if isinstance(optimizer, str):
        opt_cls = getattr(torch.optim, optimizer)
        opt = opt_cls(model.parameters(), lr=lr, **opt_kwargs)
    else:
        opt = optimizer

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class EpochIterator():
        def __init__(self, iter):
            self.iter = iter

        def __len__(self):
            return len(dataloader)

        def __iter__(self):
            return self.iter

    # for epoch in range(num_epochs):
    #     yield training_epoch(model, dataloader, opt, loss_fn, train_callback=train_callback)
    for epoch in range(num_epochs):
        epoch_iter = iter(
            training_epoch(model, dataloader, opt, loss_fn, reconstruction, train_callback=train_callback,
                           device=device))
        yield EpochIterator(epoch_iter)
