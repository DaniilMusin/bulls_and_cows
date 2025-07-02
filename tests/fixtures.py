"""Мини‑родословная для юнит‑тестов."""
import pandas as pd

pedigree = pd.DataFrame(
    [
        {"id": "G1", "mother_id": None, "father_id": None},
        {"id": "P1", "mother_id": "G1", "father_id": None},
        {"id": "P2", "mother_id": "G1", "father_id": None},
        {"id": "A",  "mother_id": "P1", "father_id": None},
        {"id": "B",  "mother_id": "P2", "father_id": None},
    ]
)
