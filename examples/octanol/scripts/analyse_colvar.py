#!/usr/bin/env python3

"""COLVAR diagnostics for the octanol MetaTally example.

Plots zeta(t), cumulative state coverage, and phi_i histograms.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ANGLE_TICKS = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
ANGLE_LABELS = [
    r"$-\pi$", r"$-2\pi/3$", r"$-\pi/3$", "0",
    r"$\pi/3$", r"$2\pi/3$", r"$\pi$"
]


def read_colvar(path):
    names = None

    with open(path) as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                names = line.split()[2:]
                break

    df = pd.read_csv(path, sep=r"\s+", comment="#", names=names)

    for col in df.columns:
        if col.startswith("phi"):
            df[col] = df[col].astype(str).str.replace("-pipi", "", regex=False).astype(float)

    return df


def cumulative_coverage(states, n_states):
    seen = set()
    coverage = np.empty(len(states), dtype=float)

    for i, state in enumerate(states):
        seen.add(int(state))
        coverage[i] = len(seen) / n_states

    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("colvar", nargs="?", default="COLVAR")
    parser.add_argument("--radix", type=int, default=3)
    parser.add_argument("--n-variables", type=int, default=7)
    parser.add_argument("--time-max", type=float)
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_colvar(args.colvar)

    if args.time_max is not None:
        df = df[df["time"] <= args.time_max]

    n_states = args.radix ** args.n_variables
    states = np.rint(df["zeta"].to_numpy()).astype(int)
    states = np.clip(states, 0, n_states - 1)
    coverage = cumulative_coverage(states, n_states)

    time = df["time"].to_numpy()

    plt.figure()
    plt.plot(time, df["zeta"], lw=1)
    plt.xlabel("Time (ps)")
    plt.ylabel(r"$\zeta$")
    plt.tight_layout()
    plt.savefig(outdir / "zeta_time.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(time, coverage, drawstyle="steps-post", lw=1.5)
    plt.xlabel("Time (ps)")
    plt.ylabel("Fraction of states visited")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(outdir / "state_coverage.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogx(time, coverage, drawstyle="steps-post", lw=1.5)
    plt.xlabel("Time (ps)")
    plt.ylabel("Fraction of states visited")
    plt.xlim(0.1, np.max (time))
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(outdir / "state_coverage_log.png", dpi=300)
    plt.close()

    for i in range(1, args.n_variables + 1):
        col = f"phi{i}"

        if col not in df:
            continue

        plt.figure()
        plt.hist(df[col], bins=60, density=True)
        plt.xlabel(rf"$\phi_{i}$ (rad)")
        plt.ylabel("Probability density")
        plt.xticks(ANGLE_TICKS, ANGLE_LABELS)
        plt.tight_layout()
        plt.savefig(outdir / f"{col}_hist.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
