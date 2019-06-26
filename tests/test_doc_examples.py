import pytest


@pytest.mark.slow
def test_readme():
    from simple_benchmark import benchmark
    import numpy as np
    funcs = [sum, np.sum]
    arguments = {i: [1] * i for i in [1, 10, 100, 1000, 10000, 100000]}
    argument_name = 'list size'
    aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
    b = benchmark(funcs, arguments, argument_name, function_aliases=aliases)
    b.to_pandas_dataframe()
    b.plot()


@pytest.mark.slow
def test_extended_benchmarkbuilder():
    from simple_benchmark import BenchmarkBuilder
    import math

    bench = BenchmarkBuilder()

    @bench.add_function()
    def sum_using_loop(lst):
        sum_ = 0
        for item in lst:
            sum_ += item
        return sum_

    @bench.add_function()
    def sum_using_range_loop(lst):
        sum_ = 0
        for idx in range(len(lst)):
            sum_ += lst[idx]
        return sum_

    bench.use_random_lists_as_arguments(sizes=[2**i for i in range(2, 15)])

    bench.add_functions([sum, math.fsum])

    b = bench.run()
    b.plot()


@pytest.mark.slow
def test_extended_multiargument():
    from itertools import starmap
    from operator import add
    from random import random

    from simple_benchmark import BenchmarkBuilder, MultiArgument

    bench = BenchmarkBuilder()

    @bench.add_function()
    def list_addition_zip(list1, list2):
        res = []
        for item1, item2 in zip(list1, list2):
            res.append(item1 + item2)
        return res

    @bench.add_function()
    def list_addition_index(list1, list2):
        res = []
        for idx in range(len(list1)):
            res.append(list1[idx] + list2[idx])
        return res

    @bench.add_function()
    def list_addition_map_zip(list1, list2):
        return list(starmap(add, zip(list1, list2)))

    @bench.add_arguments(name='list sizes')
    def benchmark_arguments():
        for size_exponent in range(2, 15):
            size = 2**size_exponent
            arguments = MultiArgument([
                [random() for _ in range(size)],
                [random() for _ in range(size)]])
            yield size, arguments

    b = bench.run()
    b.plot()


def test_extended_assert_1():
    import operator
    import random
    from simple_benchmark import assert_same_results

    funcs = [min, max]  # will produce different results
    arguments = {2**i: [random.random() for _ in range(2**i)] for i in range(2, 10)}
    with pytest.raises(AssertionError):
        assert_same_results(funcs, arguments, equality_func=operator.eq)


def test_extended_assert_2():
    import operator
    import random
    from simple_benchmark import assert_not_mutating_input

    def sort(l):
        l.sort()  # modifies the input
        return l

    funcs = [sorted, sort]
    arguments = {2**i: [random.random() for _ in range(2**i)] for i in range(2, 10)}
    with pytest.raises(AssertionError):
        assert_not_mutating_input(funcs, arguments, equality_func=operator.eq)


@pytest.mark.slow
def test_extended_time_and_max():
    from simple_benchmark import benchmark
    from datetime import timedelta

    def O_n(n):
        for i in range(n):
            pass

    def O_n_squared(n):
        for i in range(n ** 2):
            pass

    def O_n_cube(n):
        for i in range(n ** 3):
            pass

    b = benchmark(
        [O_n, O_n_squared, O_n_cube],
        {2**i: 2**i for i in range(2, 15)},
        time_per_benchmark=timedelta(milliseconds=500),
        maximum_time=timedelta(milliseconds=500)
    )

    b.plot()
