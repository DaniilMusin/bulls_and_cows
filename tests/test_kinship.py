import numpy as np
from src.kinship import make_inbreeding_fn
from .fixtures import pedigree


def test_wright_relationship():
    F, R = make_inbreeding_fn(
        pedigree.set_index("id")[["mother_id", "father_id"]].to_dict("index")  # type: ignore
    )
    # полусибсы: R = 0.25
    assert np.isclose(R("P1", "P2"), 0.25)
    # внуки общ. предка через разных полусибсов: R = 0.0625
    assert np.isclose(R("A", "B"), 0.0625)
