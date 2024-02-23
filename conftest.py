import os

import pytest

from pytensor.gradient import acc_cov, grad_cov, rop_cov, ver_cov
from pytensor.tensor.fourier import mak_cov


def pytest_sessionstart(session):
    os.environ["PYTENSOR_FLAGS"] = ",".join(
        [
            os.environ.setdefault("PYTENSOR_FLAGS", ""),
            "warn__ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,cmodule__warn_no_version=True",
        ]
    )
    os.environ["NUMBA_BOUNDSCHECK"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def print_coverage(coverage_tracker, name):
    for k, v in coverage_tracker.items():
        if v:
            print(f'{name} branch {k} is tested')
        else:
            print(f'{name} branch {k} is not tested')
    _sum = sum(coverage_tracker.values())
    _len = len(coverage_tracker)
    print(f'{name} total coverage: {_sum} out of {_len} branches = {_sum/_len:.2f}')


def pytest_sessionfinish(session, exitstatus):
    funcs = [
        [rop_cov, 'rop'],
        [grad_cov, 'grad'],
        [acc_cov, 'access_term_cache'],
        [ver_cov, 'verify_grad'],
        [mak_cov, 'make_node'],
    ]
    for coverage in funcs:
        print_coverage(*coverage)

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
