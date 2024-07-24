from collections import OrderedDict
from pathlib import Path

import argparse

import yaml

import numpy as np
import pandas as pd


aliases = {
    "algorithm": "method",
    "binary activations": "binary-act",
    "binary weights": "binary-weights"
}


def convert_val(val):
    if isinstance(val, (pd.BooleanDtype, np.bool_, bool)):
        return bool(val)
    elif isinstance(val, (pd.Float32Dtype, pd.Float64Dtype, np.float_, float)):
        return float(val)
    elif isinstance(val, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                          pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
                          np.int_, np.int64, int)):
        return int(val)
    return val


def algorithm_to_method(val):
    if isinstance(val, str):
        if val == "algorithm":
            return "method"
        return val
    elif isinstance(val, list):
        return [algorithm_to_method(v) for v in val]
    elif isinstance(val, tuple):
        return tuple([algorithm_to_method(v) for v in val])
    raise ValueError("unknown type")


ap = argparse.ArgumentParser()
ap.add_argument("INPUT", type=Path)
ap.add_argument("--by", nargs="+", default=["model", "algorithm", "binary-act", "binary-weights"])
ap.add_argument("--targets", nargs="+", required=True)
ap.add_argument("--out", type=Path, default=Path("configs"))
ap.add_argument("--metric", type=str, default="validation/accuracy")
ap.add_argument("--minimize", action="store_true")

args = ap.parse_args()
by = algorithm_to_method(args.by)
out_path: Path = args.out

input_path: Path = args.INPUT
if not input_path.exists():
    raise ValueError(f"input file does not exist: {input_path.absolute()}")

if out_path is not None:
    out_path.mkdir(exist_ok=True)

df = pd.read_csv(input_path)

df = df.rename(aliases, axis="columns")

table = df.groupby("id").min() if args.minimize else df.groupby("id").max()

print(table)

for name, group in table.groupby(by):
    print("handling group", name)
    sorted_group = group.sort_values(args.metric, ascending=args.minimize)
    row = sorted_group.iloc[0]
    targets = row[args.targets]
    filename = "_".join([f"{attr}={val}" for attr, val in zip(by, name)]) + ".csv"
    filepath = out_path / filename

    d = OrderedDict()
    for attr, val in zip(by, name):
        d[attr] = convert_val(val)
    for attr, val in targets.items():
        d[attr] = convert_val(val)

    with open(filepath, mode="w") as f:
        yaml.dump(dict(d), f)

    pass