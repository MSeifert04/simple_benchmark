import simple_benchmark

import collections


def test_simple():
    simple_benchmark.benchmark(
        funcs=[min, max],
        arguments=collections.OrderedDict([(n, [1]*n) for n in [3, 4, 5, 6]])
    )
