import os
import signal
import time

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

stop = False


def handler(signum, frame):
    global stop
    stop = True
    print("SIGINT received, interrupting execution")


signal.signal(signal.SIGINT, handler)

for i in range(args.repeat):
    for config_path in args.config:
        with open(config_path, mode="r") as f:
            config = yaml.safe_load(f)

            config_args = " ".join([format_arg(arg, val) for arg, val in config.items()])

            cmd = f"{PYTHON} {SCRIPT} {config_args}" + (" " + " ".join(args.args) if args.args else "")
            print(f"executing \"{cmd}\"")
            exitcode = os.system(cmd)
            print(f"exitcode is {exitcode}")
            print()
            pass
        print("end of iteration, waiting 2s")
        time.sleep(2)
        if stop:
            print("stopping...")
            break

print("done.")
