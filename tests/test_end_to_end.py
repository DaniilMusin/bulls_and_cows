import pandas as pd
from pathlib import Path

from src.model import assign_pairs


def test_end_to_end(tmp_path: Path):
    # небольшие фейковые данные
    (tmp_path / "bulls.csv").write_text("id,ebv\nB1,120\nB2,110\n")
    (tmp_path / "cows.csv").write_text(
        "id,ebv\nC1,100\nC2,95\nC3,105\nC4,90\nC5,98\nC6,92\n"
    )
    (tmp_path / "pedigree.csv").write_text("id,mother_id,father_id\n")

    df = assign_pairs(str(tmp_path), solver="greedy", bull_quota_frac=0.5)
    # 6 коров → 6 строк
    assert len(df) == 6
    # квота: не более 3 коров на быка
    assert max(df["bull_id"].value_counts()) <= 3

    # проверяем MILP-решатель c учётом разброса EBV
    df2 = assign_pairs(
        str(tmp_path), solver="milp", bull_quota_frac=0.5, lambda_variance=0.001
    )
    assert len(df2) == 6
    assert max(df2["bull_id"].value_counts()) <= 3

