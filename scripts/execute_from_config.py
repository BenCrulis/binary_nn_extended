import os

from pathlib import Path

import argparse

import yaml

PYTHON = "python"
SCRIPT = "train.py"


def format_arg(arg, val):
    if isinstance(val, bool):
        if val:
            return f"--{arg}"
        return ""
    return f"--{arg}={val}"


ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, nargs="+", required=True)
ap.add_argument("-n", "--repeat", type=int, default=1)
ap.add_argument("args", nargs="*")

args = ap.parse_args()

for i in range(args.repeat):
    for config_path in args.config:
        with open(config_path, mode="r") as f:
            config = yaml.safe_load(f)

            config_args = " ".join([format_arg(arg, val) for arg, val in config.items()])

            cmd = f"{PYTHON} {SCRIPT} {config_args}" + (" " + " ".join(args.args) if args.args else "")
            print(f"executing \"{cmd}\"")
            os.system(cmd)
            print()
            pass
