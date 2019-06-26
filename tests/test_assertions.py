import simple_benchmark

import operator

import numpy as np
import pytest


def sort_in_place(l):
    l.sort()
    return l


def test_assert_same_results_work():
    simple_benchmark.assert_same_results(
        funcs=[min, np.min],
        arguments={2**i: list(range(2**i)) for i in range(2, 5)},
        equality_func=operator.eq
    )


def test_assert_same_results_work_when_not_equal():
    with pytest.raises(AssertionError):
        simple_benchmark.assert_same_results(
            funcs=[min, max],
            arguments={2**i: list(range(2**i)) for i in range(2, 5)},
            equality_func=operator.eq
        )


def test_assert_not_mutating_input_work():
    simple_benchmark.assert_not_mutating_input(
        funcs=[min, np.min],
        arguments={2**i: list(range(2**i)) for i in range(2, 5)},
        equality_func=operator.eq
    )


def test_assert_not_mutating_input_work_when_modifies():
    with pytest.raises(AssertionError):
        simple_benchmark.assert_not_mutating_input(
            funcs=[sorted, sort_in_place],
            arguments={2**i: list(reversed(range(2**i))) for i in range(2, 5)},
            equality_func=operator.eq
        )
