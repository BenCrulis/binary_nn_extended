import argparse
from pathlib import Path

import pandas as pd

import wandb

from tqdm import tqdm

import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ENTITY_PROJECT_ID", help="<entity>/<project>")
    parser.add_argument("filename", type=Path, help="export into this file")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--timeout", type=int, default=60, help="timeout for graphql requests")

    return parser.parse_args()


def main():
    args = parse_args()

    ent_proj = args.ENTITY_PROJECT_ID
    filename = args.filename
    is_sweep = args.sweep
    timeout = args.timeout
    api = wandb.Api(timeout=timeout)

    print(f"starting export of {ent_proj}.")

    if is_sweep:
        sweep = api.sweep(ent_proj)
        runs = sweep.runs
    else:
        runs = api.runs(ent_proj)

    df = None

    dfs = []
    for run in tqdm(runs):
        history: pd.DataFrame = run.history(samples=1e9)
        history["run_id"] = run.id

        for col, val in run.config.items():
            history[col] = val

        summary = run.summary

        for col, val in summary.items():
            curval = None
            if col in history.columns:
                curval = history[col].loc[0]
            else:
                history[col] = None
            if curval is None or curval == "" or (isinstance(curval, float) and math.isnan(curval)):
                history.iloc[0, history.columns.get_loc(col)] = val if isinstance(val, (float, int, str, bool)) else str(val)

        dfs.append(history)

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
