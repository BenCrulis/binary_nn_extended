import torch

import torchmetrics
from torch.utils.data import DataLoader


def eval_classification_iterator(model, ds, metrics, batch_size=100):
    it = iter(eval_classification(model, ds, metrics, batch_size))

    class EvalIterator():
        def __len__(self):
            return len(ds) // batch_size

        def __iter__(self):
            return it
    return EvalIterator()


def eval_classification(model, ds, metrics, batch_size=100):

    dl = DataLoader(ds, batch_size=batch_size)

    for x, y in dl:
        with torch.inference_mode():
            model.eval()
            y_pred = model(x)

            res = {}
            for k, metric in metrics.items():
                metric_v = metric(y_pred, y)
                res[k] = metric_v

            yield res
