import numpy as np
import pytest

from segtrack import rle


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((37, 53)) > 0.6
    assert np.array_equal(rle.decode(rle.encode(mask), 37, 53), mask)


def test_edge_cases():
    empty = np.zeros((4, 5), bool)
    full = np.ones((4, 5), bool)
    assert rle.encode(empty).tolist() == [20]
    assert rle.encode(full).tolist() == [0, 20]
    for m in (empty, full):
        assert np.array_equal(rle.decode(rle.encode(m), 4, 5), m)


def test_column_major_order():
    mask = np.array([[1, 0], [0, 0]], bool)          # only top-left set
    assert rle.encode(mask).tolist() == [0, 1, 3]
    mask = np.array([[0, 1], [0, 0]], bool)          # top-right is 3rd in column-major order
    assert rle.encode(mask).tolist() == [2, 1, 1]
