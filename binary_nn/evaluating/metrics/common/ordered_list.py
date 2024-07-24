import torch
import wandb


def ordered_list(values, name_template):
    return {name_template.format(i=i, val=val): val for i, val in enumerate(values)}


def wandb_table_layers(values, col_name):
    return wandb.Table([col_name, "layer"], data=torch.stack([values, torch.arange(len(values))]).T)
