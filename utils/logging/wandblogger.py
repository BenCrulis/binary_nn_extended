import wandb

from utils.logging.logger import Logger


class WandbLogger(Logger):
    def __init__(self, *args, **kwargs):
        wandb.init(*args, **kwargs)

    def log(self, logs, step=None, commit=None):
        if commit is None:
            commit = step is None
        wandb.log(logs, step=step, commit=commit)

