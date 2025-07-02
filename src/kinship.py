"""
Инбридинг (F) и коэффициент родства (R) по Райту, 1922.

R(a,b) = 2 · f(a,b), где f – коэффициент коанцестри.
Алгоритм рекурсивный с мемоизацией + numba‑ускоренная фильтрация
допустимых пар bulls × cows.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numba import njit

PedigreeType = Dict[str, Tuple[str | None, str | None]]  # id → (mother, father)


def make_inbreeding_fn(ped_dict: PedigreeType):
    """
    Возвращает две функции:
        F(id) → inbreeding
        R(a, b) → relationship (родство, 0…2)
    """
    def _parents(animal_id: str):
        val = ped_dict.get(animal_id, (None, None))
        if isinstance(val, dict):
            return val.get("mother_id"), val.get("father_id")
        return val

    @lru_cache(maxsize=None)
    def _f(a: str, b: str) -> float:
        # коэффициент коанцестри f(a,b)
        if a is None or b is None:
            return 0.0
        if a == b:
            return 0.5 * (1 + F(a))
        ma, fa = _parents(a)
        mb, fb = _parents(b)
        s = 0.0
        for pa in (ma, fa):
            for pb in (mb, fb):
                if pa is not None and pb is not None:
                    s += 0.25 * _f(pa, pb)
        return s

    @lru_cache(maxsize=None)
    def F(animal_id: str) -> float:
        mother, father = _parents(animal_id)
        if mother is None or father is None:
            return 0.0
        return _f(mother, father)

    def R(a: str, b: str) -> float:
        return 2.0 * _f(a, b)

    return F, R


@njit(cache=True)
def _filter_pairs_numba(
    kinship_mat: np.ndarray,
    ebv_bulls: np.ndarray,
    ebv_cows: np.ndarray,
    r_threshold: float,
):
    rows_cow = []
    rows_bull = []
    rows_ebv = []
    nbulls, ncows = kinship_mat.shape
    for j in range(nbulls):
        for i in range(ncows):
            if kinship_mat[j, i] <= r_threshold:
                rows_cow.append(i)
                rows_bull.append(j)
                rows_ebv.append(0.5 * (ebv_bulls[j] + ebv_cows[i]))
    return (
        np.asarray(rows_cow, dtype=np.int32),
        np.asarray(rows_bull, dtype=np.int32),
        np.asarray(rows_ebv, dtype=np.float64),
    )


def build_feasible_pairs(
    bulls: pd.DataFrame,
    cows: pd.DataFrame,
    R_fn,
    r_threshold: float = 0.05,
) -> pd.DataFrame:
    """Формирует таблицу допустимых пар (cow_id, bull_id, ebv_child)."""
    bull_ids = bulls["id"].to_numpy(dtype=str)
    cow_ids = cows["id"].to_numpy(dtype=str)
    ebv_bulls = bulls["ebv"].to_numpy(dtype=np.float64)
    ebv_cows = cows["ebv"].to_numpy(dtype=np.float64)

    # матрица родства bulls × cows
    kinship_mat = np.zeros((bull_ids.shape[0], cow_ids.shape[0]), dtype=np.float32)
    for j, bid in enumerate(bull_ids):
        for i, cid in enumerate(cow_ids):
            kinship_mat[j, i] = R_fn(bid, cid)

    cow_idx, bull_idx, ebv_child = _filter_pairs_numba(
        kinship_mat, ebv_bulls, ebv_cows, r_threshold
    )

    return pd.DataFrame(
        {
            "cow_id": cow_ids[cow_idx],
            "bull_id": bull_ids[bull_idx],
            "ebv_child": ebv_child,
        }
    )
