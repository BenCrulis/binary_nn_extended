import argparse
from pathlib import Path

import tempfile

import pandas as pd

import wandb

import json

from tqdm import tqdm

import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ENTITY_PROJECT_ID", help="<entity>/<project>")
    parser.add_argument("filename", type=Path, help="export into this file")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--timeout", type=int, default=60, help="timeout for graphql requests")
    parser.add_argument("--include", nargs="*", help="metrics columns to include")
    parser.add_argument("--tag-filter", type=str, help="keep runs with tag")

    return parser.parse_args()


def main():
    args = parse_args()

    ent_proj = args.ENTITY_PROJECT_ID
    filename: Path = args.filename
    is_sweep = args.sweep
    timeout = args.timeout
    tag = args.tag_filter
    cols = None if args.include is None else (["_timestamp", "_step", "_runtime"] + args.include)
    api = wandb.Api(timeout=timeout)

    assert filename.parent.exists()

    print(f"starting export of {ent_proj}.")

    if is_sweep:
        sweep = api.sweep(ent_proj)
        runs = sweep.runs
    else:
        runs = api.runs(ent_proj)

    df = None

    tempdir = Path(tempfile.gettempdir())
    metadata_path = tempdir / "wandb-metadata.json"

    dfs = []
    for run in tqdm(runs):
        if tag is not None and tag not in run.tags:
            continue
        history: pd.DataFrame = run.history(samples=1e9)
        history["run_id"] = run.id

        for col, val in run.config.items():
            history[col] = val

        summary = run.summary
        systemMetrics = run.systemMetrics
        metadata_file = run.file("wandb-metadata.json")
        metadata_file.download(root=tempdir, replace=True)
        with open(metadata_path) as f:
            metadata = json.load(f)

        if cols is not None:
            history = history[[col for col in cols if col in history.columns]]

        for col, val in summary.items():
            if col is not None and cols is not None and col not in cols:  # don't iterate on columns we won't save anyways
                continue
            curval = None
            if col in history.columns:
                curval = history[col].loc[0]
            else:
                history[col] = None
            if curval is None or curval == "" or (isinstance(curval, float) and math.isnan(curval)):
                history.iloc[0, history.columns.get_loc(col)] = val if isinstance(val, (float, int, str, bool)) else str(val)

        gpu_name = metadata["gpu"]
        history.insert(loc=0, column="id", value=run.id)
        history.insert(loc=1, column="gpu", value=gpu_name)
        history.insert(loc=1, column="host", value=metadata["host"])

        gpu_total_mem = [x["memory_total"] for x in metadata["gpu_devices"] if x["name"] == gpu_name][0]
        gpu_index = 1 if "system.gpu.process.0.powerWatts" not in systemMetrics.keys() else 0
        watts = systemMetrics[f"system.gpu.process.{gpu_index}.powerWatts"]
        # gpu_time_access_mem_percent = systemMetrics["system.gpu.process.0.memory"]
        gpu_mem_alloc_percent = systemMetrics[f"system.gpu.process.{gpu_index}.memoryAllocated"]

        history.insert(loc=3, column="gpu_watts", value=watts)
        history.insert(loc=3, column="gpu_mem_alloc_percent", value=gpu_mem_alloc_percent)
        history.insert(loc=3, column="gpu_total_mem", value=gpu_total_mem)

        dfs.append(history)

    if metadata_path.exists():
        metadata_path.unlink()

    df = pd.concat(dfs)
    if filename.exists():
        df.to_csv(filename, mode="a", index=False, header=False)
    else:
        print(f"creating {filename}")
        df.to_csv(filename, mode="w", index=False, header=True)

        pass
    print(f"finished export of {ent_proj}.")


if __name__ == '__main__':
    main()
