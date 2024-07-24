import torch
import wandb


def ordered_list(values, name_template):
    return {name_template.format(i=i, val=val): val for i, val in enumerate(values)}


def wandb_table_layers(values, col_name):
    return wandb.Table(["layer", col_name], data=[[i, val] for i, val in enumerate(values)])
