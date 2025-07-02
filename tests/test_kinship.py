import numpy as np
from src.kinship import make_inbreeding_fn, build_additive_matrix
from .fixtures import pedigree


def test_wright_relationship():
    F, R = make_inbreeding_fn(
        pedigree.set_index("id")[["mother_id", "father_id"]].to_dict("index")  # type: ignore
    )
    # полусибсы: R = 0.25
    assert np.isclose(R("P1", "P2"), 0.25)
    # внуки общ. предка через разных полусибсов: R = 0.0625
    assert np.isclose(R("A", "B"), 0.0625)


def test_additive_matrix():
    A = build_additive_matrix(pedigree)
    assert np.isclose(A.loc["P1", "P2"], 0.25)
    assert np.isclose(A.loc["A", "B"], 0.0625)
    # диагональ = 1 + F; для базовых животных F=0
    assert np.isclose(A.loc["G1", "G1"], 1.0)
