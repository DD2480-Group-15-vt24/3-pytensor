import os.path as path
from tempfile import mkdtemp

import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.tensor.type import (
    TensorType,
)


filter_covarage = {
    83: False,  # Branch 83
    84: False,  # Branch 84
    85: False,  # Branch 85
    86: False,  # Branch 86
    87: False,  # Branch 87
    88: False,  # Branch 88
    89: False,  # Branch 89
    90: False,  # Branch 90
    91: False,  # Branch 91
    92: False,  # Branch 92
    93: False,  # Branch 93
    94: False,  # Branch 94
    95: False,  # Branch 95
    96: False,  # Branch 96
    97: False,  # Branch 97
    98: False,  # Branch 98
    99: False,  # Branch 99
    100: False,  # Branch 100
}


def calculate_percentage():
    true_count = sum(1 for value in filter_covarage.values() if value)
    return (true_count / len(filter_covarage)) * 100


# original tests, test #83-93 and 97 true --> 66%
def test_filter_variable():
    test_type = TensorType(config.floatX, shape=())

    with pytest.raises(TypeError):
        test_type.filter(test_type(), filter_covarage)

    test_type = TensorType(config.floatX, shape=(1, None))

    with pytest.raises(TypeError):
        test_type.filter(np.empty((0, 1), dtype=config.floatX), filter_covarage)

    with pytest.raises(TypeError, match=".*not aligned.*"):
        test_val = np.empty((1, 2), dtype=config.floatX)
        test_val.flags.aligned = False
        test_type.filter(test_val, filter_covarage)

    with pytest.raises(ValueError, match="Non-finite"):
        test_type.filter_checks_isfinite = True
        test_type.filter(np.full((1, 2), np.inf, dtype=config.floatX), filter_covarage)

    test_type2 = TensorType(config.floatX, shape=(None, None))
    test_var = test_type()
    test_var2 = test_type2()

    res = test_type.filter_variable(test_var, allow_convert=True)
    assert res is test_var

    # Make sure it returns the more specific type
    res = test_type.filter_variable(test_var2, allow_convert=True)
    assert res.type == test_type

    test_type3 = TensorType(config.floatX, shape=(1, 20))
    res = test_type3.filter_variable(test_var, allow_convert=True)
    assert res.type == test_type3


def test_filter_strict():
    test_type = TensorType(config.floatX, shape=())

    with pytest.raises(TypeError):
        test_type.filter(1, filter_covarage, strict=True)

    with pytest.raises(TypeError):
        test_type.filter(np.array(1, dtype=int), filter_covarage, strict=True)


def test_filter_ndarray_subclass():
    """Make sure `TensorType.filter` can handle NumPy `ndarray` subclasses."""
    test_type = TensorType(config.floatX, shape=(None,))

    class MyNdarray(np.ndarray):
        pass

    test_val = np.array([1.0], dtype=config.floatX).view(MyNdarray)
    assert isinstance(test_val, MyNdarray)

    res = test_type.filter(test_val, filter_covarage)
    assert isinstance(res, MyNdarray)
    assert res is test_val


def test_filter_float_subclass():
    """Make sure `TensorType.filter` can handle `float` subclasses."""
    with config.change_flags(floatX="float64"):
        test_type = TensorType("float64", shape=())

        nan = np.array([np.nan], dtype="float64")[0]
        assert isinstance(nan, float) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan, filter_covarage)
        assert isinstance(filtered_nan, np.ndarray)

    with config.change_flags(floatX="float32"):
        # Try again, except this time `nan` isn't a `float`
        test_type = TensorType("float32", shape=())

        nan = np.array([np.nan], dtype="float32")[0]
        assert isinstance(nan, np.floating) and not isinstance(nan, np.ndarray)

        filtered_nan = test_type.filter(nan, filter_covarage)
        assert isinstance(filtered_nan, np.ndarray)


def test_filter_memmap():
    r"""Make sure `TensorType.filter` can handle NumPy `memmap`\s subclasses."""
    data = np.arange(12, dtype=config.floatX)
    data.resize((3, 4))
    filename = path.join(mkdtemp(), "newfile.dat")
    fp = np.memmap(filename, dtype=config.floatX, mode="w+", shape=(3, 4))

    test_type = TensorType(config.floatX, shape=(None, None))

    res = test_type.filter(fp, filter_covarage)
    assert res is fp


# New test 4, branch #94, 95 and 96 true --> 83%
def test_up_dtype():
    test_type = TensorType(config.floatX, shape=())
    data = np.array([1, 2, 3], dtype="int32")
    with pytest.raises(TypeError):
        test_type.filter(data, filter_covarage)


def test_print_coverage():
    print(calculate_percentage(), "%")
    print(filter_covarage)
    assert 1 == 1
