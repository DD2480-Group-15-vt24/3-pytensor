import numpy as np
import pytest

from pytensor.compile.function import function
from pytensor.compile.mode import get_default_mode
from pytensor.compile.sharedvalue import shared
from pytensor.configdefaults import config
from pytensor.tensor.math import (
    mean,
)
from pytensor.tensor.type import (
    vector,
)
from tests.tensor.utils import (
    random,
)


if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()

mean_coverage = {
    68: False,  # Branch 68
    69: False,  # Branch 69
    70: False,  # Branch 70
    71: False,  # Branch 71
    72: False,  # Branch 72
    73: False,  # Branch 73
    74: False,  # Branch 74
    75: False,  # Branch 75
    76: False,  # Branch 76
    77: False,  # Branch 77
    78: False,  # Branch 78
    79: False,  # Branch 79
    80: False,  # Branch 80
    81: False,  # Branch 81
    82: False,  # Branch 82
}


def calculate_percentage():
    true_count = sum(1 for value in mean_coverage.values() if value)
    return (true_count / len(mean_coverage)) * 100


# original tests, branch #68, 72, 73, 74, 75, 76, 77,81 and 82 true --> 60%
def test_mean_single_element():
    res = mean(np.zeros(1), mean_coverage)
    assert res.eval() == 0.0


def test_basic():
    x = vector()
    f = function([x], mean(x, mean_coverage))
    data = random(50)
    assert np.allclose(f(data), np.mean(data))


def test_list():
    ll = [shared(0.0), shared(2.0)]
    assert mean(ll, mean_coverage).eval() == 1


# New test 1, branch #69 --> 66,6666%
def test_dtype_arg():
    with pytest.raises(NotImplementedError):
        mean(np.zeros(1), mean_coverage, op=True, dtype="float32")


# New test 2, branch #70 --> 73,33333%
def test_acc_dtype_arg():
    with pytest.raises(NotImplementedError):
        mean(np.zeros(1), mean_coverage, op=True, acc_dtype="float32")


# New test 3, branch #71 --> 80%
def test_acc_dtype_arg_2():
    ll = [shared(0.0), shared(2.0)]
    assert mean(ll, mean_coverage, axis=0, keepdims=True, op=True).eval() == 1


def test_print_coverage():
    print(calculate_percentage(), "%")
    print(mean_coverage)
    assert 1 == 1
