"""
Два решателя:
    * greedy – 10-15 s на полном датасете
    * milp   – 4-8 мин (PuLP + CBC)
"""
from __future__ import annotations
import logging
import math
from typing import Literal

import pandas as pd
import pulp
from tqdm import tqdm

from .kinship import build_feasible_pairs, build_additive_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_data(data_dir: str):
    bulls = pd.read_csv(f"{data_dir}/bulls.csv").dropna(subset=["ebv"])
    cows = pd.read_csv(f"{data_dir}/cows.csv").dropna(subset=["ebv"])
    pedigree = pd.read_csv(f"{data_dir}/pedigree.csv")
    return bulls, cows, pedigree


def assign_pairs(
    data_dir: str,
    solver: Literal["greedy", "milp"] = "greedy",
    r_threshold: float = 0.05,
    bull_quota_frac: float = 0.10,
    lambda_variance: float | None = None,
) -> pd.DataFrame:
    LOGGER.info("📦  Loading data …")
    bulls, cows, pedigree = _load_data(data_dir)

    LOGGER.info("🔍  Building additive kinship matrix …")
    ids = list(bulls["id"]) + list(cows["id"])
    A_mat = build_additive_matrix(pedigree, ids)

    LOGGER.info("🔍  Building feasible (bull, cow) pairs …")
    pairs = build_feasible_pairs(bulls, cows, r_threshold=r_threshold, A_mat=A_mat)

    if solver == "greedy":
        return _solve_greedy(pairs, bulls, cows, bull_quota_frac)
    elif solver == "milp":
        if lambda_variance is None:
            return _solve_milp(pairs, bulls, cows, bull_quota_frac)
        return _solve_milp(pairs, bulls, cows, bull_quota_frac, lambda_variance)
    else:
        raise ValueError(f"Unknown solver: {solver!r}")


# --------------------------------------------------------------------------- #
# 1. Greedy
# --------------------------------------------------------------------------- #
def _solve_greedy(
    pairs: pd.DataFrame,
    bulls: pd.DataFrame,
    cows: pd.DataFrame,
    bull_quota_frac: float,
) -> pd.DataFrame:
    LOGGER.info("⚡  Greedy solver …")
    quota = math.ceil(bull_quota_frac * len(cows))
    bull_load = {b: 0 for b in bulls["id"]}
    # dict cow → [(bull, ebv_child)…]  (сорт EBV desc)
    pairs_sorted = (
        pairs.sort_values("ebv_child", ascending=False)
        .groupby("cow_id")
        .apply(
            lambda df: list(
                df[["bull_id", "ebv_child"]].itertuples(index=False, name=None)
            )
        )
        .to_dict()
    )

    assignment = {}
    for cow_id in cows.sort_values("ebv", ascending=False)["id"]:
        for bull_id, _ in pairs_sorted.get(cow_id, []):
            if bull_load[bull_id] < quota:
                assignment[cow_id] = bull_id
                bull_load[bull_id] += 1
                break
        else:
            raise RuntimeError(f"No feasible bull found for cow {cow_id}")

    return pd.DataFrame({"cow_id": assignment.keys(), "bull_id": assignment.values()})


# --------------------------------------------------------------------------- #
# 2. MILP
# --------------------------------------------------------------------------- #
def _solve_milp(
    pairs: pd.DataFrame,
    bulls: pd.DataFrame,
    cows: pd.DataFrame,
    bull_quota_frac: float,
    lambda_variance: float = 1e-3,
) -> pd.DataFrame:
    LOGGER.info("🧮  Building MILP with multi-objective (Mean + Variance)...")
    quota = math.ceil(bull_quota_frac * len(cows))
    prob = pulp.LpProblem("cow_bull_assignment", pulp.LpMaximize)

    # переменные x[cow,bull] ∈ {0,1}
    x = pulp.LpVariable.dicts(
        "x",
        (tuple(r) for r in pairs[["cow_id", "bull_id"]].itertuples(index=False, name=None)),
        lowBound=0,
        upBound=1,
        cat="Binary",
    )

    # 1. Терм для суммарного EBV
    total_ebv_term = pulp.lpSum(
        pairs.apply(lambda r: r.ebv_child * x[(r.cow_id, r.bull_id)], axis=1)
    )

    # 2. Терм для разброса EBV (через сумму абсолютных отклонений)
    num_cows = len(cows)
    mean_ebv = pulp.LpVariable("mean_EBV")
    prob += mean_ebv == (1 / num_cows) * total_ebv_term

    dev_vars = pulp.LpVariable.dicts(
        "dev", (c for c in cows["id"]), lowBound=0, upBound=1e6
    )

    pairs_by_cow = pairs.groupby("cow_id")
    LOGGER.info("➕  Adding variance constraints...")
    for cow_id, group in tqdm(pairs_by_cow, desc="variance"):
        offspring_ebv = pulp.lpSum(
            r.ebv_child * x[(r.cow_id, r.bull_id)] for r in group.itertuples()
        )
        prob += offspring_ebv - mean_ebv <= dev_vars[cow_id]
        prob += mean_ebv - offspring_ebv <= dev_vars[cow_id]

    variance_term = pulp.lpSum(dev_vars[c] for c in cows["id"])

    # 3. Установка комбинированной цели
    prob += total_ebv_term + lambda_variance * variance_term

    # каждая корова – ровно за одним быком
    LOGGER.info("➕  Cow constraints …")
    for cow_id, g in tqdm(pairs.groupby("cow_id"), desc="cows"):
        prob += pulp.lpSum(x[(cow_id, b)] for b in g["bull_id"]) == 1

    # квота на быка
    LOGGER.info("➕  Bull constraints (quota %d) …", quota)
    for bull_id, g in tqdm(pairs.groupby("bull_id"), desc="bulls"):
        prob += pulp.lpSum(x[(c, bull_id)] for c in g["cow_id"]) <= quota

    LOGGER.info("🚀  Solving MILP (CBC) …")
    prob.solve(pulp.PULP_CBC_CMD(msg=True, presolve=True))

    LOGGER.info("✅  Status: %s, objective = %.3f",
                pulp.LpStatus[prob.status], pulp.value(prob.objective))

    chosen = [(c, b) for (c, b), var in x.items() if var.value() == 1]
    return pd.DataFrame(chosen, columns=["cow_id", "bull_id"])
