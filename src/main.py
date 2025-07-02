#!/usr/bin/env python3
"""
CLI‑обёртка: быстро получить cow_bull_assignments.csv.

Примеры:
    python -m src.main --data_dir data --solver greedy
    python -m src.main --data_dir data --solver milp --out result.csv
"""
from __future__ import annotations
import argparse

import pandas as pd

from .model import assign_pairs


def _parse():
    p = argparse.ArgumentParser("cow‑bull assignment")
    p.add_argument("--data_dir", default="data",
                   help="директорий с bulls.csv, cows.csv, pedigree.csv")
    p.add_argument("--solver", choices=["greedy", "milp"], default="greedy")
    p.add_argument("--out", default="cow_bull_assignments.csv")
    p.add_argument("--lambda_variance", type=float, default=None,
                   help="вес разброса EBV в MILP")

    return p.parse_args()


def main():
    args = _parse()

    df: pd.DataFrame = assign_pairs(
        args.data_dir,
        solver=args.solver,
        lambda_variance=args.lambda_variance,
    )

    df.to_csv(args.out, index=False)
    print(f"✅  Saved {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
