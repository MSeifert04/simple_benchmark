import simple_benchmark


def test_simple():
    bb = simple_benchmark.BenchmarkBuilder()
    bb.add_functions([min, max])
    bb.use_random_lists_as_arguments([2, 3, 4])
    bb.run()
