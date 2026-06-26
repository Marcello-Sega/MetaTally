#!/usr/bin/env python3

"""Count visited encoded states for the octanol MetaTally example."""

import argparse
import numpy as np
import pandas as pd


def read_colvar(path):
    names = None

    with open(path) as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                names = line.split()[2:]
                break

    return pd.read_csv(path, sep=r"\s+", comment="#", names=names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("colvar", nargs="?", default="COLVAR")
    parser.add_argument("--zeta", default="zeta")
    parser.add_argument("--radix", type=int, default=3)
    parser.add_argument("--n-variables", type=int, default=7)
    parser.add_argument("--time-max", type=float)
    args = parser.parse_args()

    df = read_colvar(args.colvar)

    if args.time_max is not None:
        df = df[df["time"] <= args.time_max]

    n_states = args.radix ** args.n_variables

    # Continuous zeta values are rounded to encoded state labels.
    states = np.rint(df[args.zeta].to_numpy()).astype(int)
    states = np.clip(states, 0, n_states - 1)

    visited = np.unique(states)
    missing = np.setdiff1d(np.arange(n_states), visited)

    print(f"Frames analysed: {len(df)}")
    print(f"Visited states: {len(visited)}")
    print(f"Total possible states: {n_states}")
    print(f"Fraction visited: {len(visited) / n_states:.6f}")
    print(f"Minimum state: {visited.min()}")
    print(f"Maximum state: {visited.max()}")
    print(f"Missing states: {len(missing)}")

    if len(missing) > 0:
        print(missing)


if __name__ == "__main__":
    main()
