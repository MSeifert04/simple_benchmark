# Licensed under Apache License Version 2.0 - see LICENSE
"""
.. warning::
   This package is under active development. API changes are very likely.

This package aims to give an easy way to benchmark several functions for
different inputs and provide ways to visualize the benchmark results.

To utilize the full features (visualization and post-processing) you need to
install the optional dependencies:

- NumPy
- Pandas
- Matplotlib
"""

__version__ = '0.0.1'

__all__ = ['benchmark', 'benchmark_random_array', 'benchmark_random_list',
           'BenchmarkResult']

import functools
import itertools
import pprint
import random
import timeit


def _estimate_number_of_repeats(func, target_seconds):
    """Estimate the number of repeats for a function so that the benchmark will take a specific time.

    In case the function is very slow or really fast some default values are returned.

    Parameters
    ----------
    func : callable
        The function to time. Must not have required arguments!
    target_seconds : float
        The amount of second the benchmark should roughly take.
        Decimal values below 1 are possible.

    Returns
    -------
    repeats : int
        The number of repeats.
    number : int
        The number of timings in each repetition.
    """
    # Just for a quick reference:
    # One millisecond is 1e-3
    # One microsecond is 1e-6
    # One nanosecond  is 1e-9
    single_time = timeit.timeit(func, number=1)

    # Get a more accurate baseline if the function was really fast
    if single_time < 1e-6:
        single_time = timeit.timeit(func, number=1000) / 1000
    if single_time < 1e-5:
        single_time = timeit.timeit(func, number=100) / 100
    elif single_time < 1e-4:
        single_time = timeit.timeit(func, number=10) / 10

    n_repeats = int(target_seconds / single_time)
    # The timeit execution should be at least 10-100us so that the granularity
    # of the timer isn't a limiting factor.
    if single_time < 1e-4:
        factor = 1e-4 / single_time
        return int(n_repeats // factor), int(factor)
    # Otherwise the number of timings each repeat should be 1.
    # However make sure there are at least 3 repeats for each function!
    return min(3, n_repeats), 1


def benchmark(
        funcs,
        arguments,
        argument_name="",
        warmups=None,
        time_per_benchmark=0.1,
        function_aliases=None):
    """Create a benchmark suite for different functions and for different arguments.

    Parameters
    ----------
    funcs : iterable of callables
        The functions to benchmark.
    arguments : dict
        A dictionary containing where the key represents the reported value
        (for example an integer representing the list size) as key and the argument
        for the functions (for example the list) as value.
    argument_name : str, optional
        The name of the reported value. For example if the arguments represent
        list sizes this could be `"size of the list"`.
        Default is an empty string.
    warmups : None or iterable of callables, optional
        If not None it specifies the callables that need a warmup call
        before being timed. That is so, that caches can be filled or
        jitters to kick in.
        Default is None.
    time_per_benchmark : float, optional
        Each benchmark should take approximately this value in seconds.
        However the value is ignored for functions that take very little time
        or very long.
        Default is 0.1 (seconds).
    function_aliases : None or dict, optional
        If not None it should be a dictionary containing the function as key
        and the name of the function as value. The value will be used in the
        final reports and plots.
        Default is None.

    Returns
    -------
    benchmark : BenchmarkResult
        The result of the benchmarks.

    Examples
    --------
    For example to benchmark different sum functions on a Python list.

    The setup::

        >>> from simple_benchmark import benchmark
        >>> import numpy as np
        >>> funcs = [sum, np.sum]
        >>> arguments = {i: [1]*i for i in [1, 10, 100, 1000, 10000, 100000]}
        >>> argument_name = 'list size'
        >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
        >>> b = benchmark(funcs, arguments, argument_name, function_aliases=aliases)

    Inspecting the results::

        >>> b.to_pandas_dataframe()

    Plotting the results::

        >>> b.plot()
        >>> b.plot(relative_to=np.sum)
        >>> b.plot_both(relative_to=sum)

    See also
    --------
    benchmark_random_array, benchmark_random_list
    """
    funcs = list(funcs)
    warm_up_calls = {func: 0 for func in funcs}
    if warmups is not None:
        for func in warmups:
            warm_up_calls[func] = 1
    function_aliases = function_aliases or {}

    timings = {func: [] for func in funcs}
    for arg in arguments.values():
        for func, timing_list in timings.items():
            bound_func = functools.partial(func, arg)
            for _ in itertools.repeat(None, times=warm_up_calls[func]):
                bound_func()
            repeats, number = _estimate_number_of_repeats(bound_func, time_per_benchmark)
            # As per the timeit module documentation a very good approximation
            # of a timing is found by repeating the benchmark and using the
            # minimum.
            times = timeit.repeat(bound_func, number=number, repeat=repeats)
            time = min(times)
            timing_list.append(time / number)
    return BenchmarkResult(timings, function_aliases, arguments, argument_name)


def benchmark_random_array(
        funcs,
        sizes,
        warmups=None,
        time_per_benchmark=0.1,
        function_aliases=None):
    """A shortcut for :func:`benchmark` if a random array is wanted.

    The arguments *arguments* and *argument_name* of the normal constructor
    are replaced with a simple *size* argument.

    Parameters
    ----------
    funcs : iterable of callables
        The functions to benchmark.
    sizes : iterable of int
        The different size values for arrays.
    warmups : None or iterable of callables, optional
        If not None it specifies the callables that need a warmup call
        before being timed. That is so, that caches can be filled or
        jitters to kick in.
        Default is None.
    time_per_benchmark : float, optional
        Each benchmark should take approximately this value in seconds.
        However the value is ignored for functions that take very little time
        or very long.
        Default is 0.1 (seconds).
    function_aliases : None or dict, optional
        If not None it should be a dictionary containing the function as key
        and the name of the function as value. The value will be used in the
        final reports and plots.
        Default is None.

    Returns
    -------
    benchmark : BenchmarkResult
        The result of the benchmarks.

    Raises
    ------
    ImportError
        If NumPy isn't installed.

    Examples
    --------

    In case the arguments are NumPy arrays containing random floats this
    function allows for a more concise benchmark::

        >>> from simple_benchmark import benchmark_random_array
        >>> import numpy as np
        >>> funcs = [sum, np.sum]
        >>> sizes = [i ** 4 for i in range(20)]
        >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
        >>> b = benchmark_random_array(funcs, sizes, function_aliases=aliases)

    See also
    --------
    benchmark, benchmark_random_list
    """

    try:
        import numpy as np
    except ImportError:
        raise ImportError('simple_benchmark requires NumPy for this function.')
    return benchmark(
        funcs,
        arguments={size: np.random.random(size) for size in sizes},
        argument_name='array size',
        warmups=warmups,
        time_per_benchmark=time_per_benchmark,
        function_aliases=function_aliases)


def benchmark_random_list(
        funcs,
        sizes,
        warmups=None,
        time_per_benchmark=0.1,
        function_aliases=None):
    """A shortcut for :func:`benchmark` if a random list is wanted.

    The arguments *arguments* and *argument_name* of the normal constructor
    are replaced with a simple *size* argument.

    Parameters
    ----------
    funcs : iterable of callables
        The functions to benchmark.
    sizes : iterable of int
        The different size values for list.
    warmups : None or iterable of callables, optional
        If not None it specifies the callables that need a warmup call
        before being timed. That is so, that caches can be filled or
        jitters to kick in.
        Default is None.
    time_per_benchmark : float, optional
        Each benchmark should take approximately this value in seconds.
        However the value is ignored for functions that take very little time
        or very long.
        Default is 0.1 (seconds).
    function_aliases : None or dict, optional
        If not None it should be a dictionary containing the function as key
        and the name of the function as value. The value will be used in the
        final reports and plots.
        Default is None.

    Returns
    -------
    benchmark : BenchmarkResult
        The result of the benchmarks.

    Examples
    --------

    In case the arguments are lists containing random floats this function
    allows for a more concise benchmark::

        >>> from simple_benchmark import benchmark_random_list
        >>> import numpy as np
        >>> funcs = [sum, np.sum]
        >>> sizes = [i ** 4 for i in range(20)]
        >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
        >>> b = benchmark_random_list(funcs, sizes, function_aliases=aliases)

    See also
    --------
    benchmark, benchmark_random_array
    """
    random_func = random.random
    return benchmark(
        funcs,
        arguments={size: [random_func() for _ in itertools.repeat(None, times=size)]
                   for size in sizes},
        argument_name='list size',
        warmups=warmups,
        time_per_benchmark=time_per_benchmark,
        function_aliases=function_aliases)


class BenchmarkResult(object):
    """A class holding a benchmarking result that provides additional printing and plotting functions."""
    def __init__(self, timings, function_aliases, arguments, argument_name):
        self._timings = timings
        self.function_aliases = function_aliases
        self._arguments = arguments
        self._argument_name = argument_name

    def __str__(self):
        try:
            return str(self.to_pandas_dataframe())
        except ImportError:
            return pprint.pformat({self._function_name(k): v for k, v in self._timings.items()})

    __repr__ = __str__

    def _function_name(self, func):
        try:
            return self.function_aliases[func]
        except KeyError:
            # Has to be a different branch because not every function has a
            # __name__ attribute. So we cannot simply use the dictionaries `get`
            # with default.
            try:
                return func.__name__
            except AttributeError:
                raise TypeError('function "func" does not have a __name__ attribute. '
                                'Please use "function_aliases" to provide a function name alias.')

    def to_pandas_dataframe(self):
        """Return the timing results as pandas Dataframe. This is the preferred
        way of accessing the text form of the timings.

        Returns
        -------
        pandas.DataFrame
            The timings as DataFrame.

        Raises
        ------
        ImportError
            If pandas isn't installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('simple_benchmark requires pandas for this method.')
        return pd.DataFrame(
            {self._function_name(func): timings for func, timings in self._timings.items()},
            index=list(self._arguments))

    def plot(self, relative_to=None, ax=None):
        """Plot the benchmarks, either relative or absolute.

        Parameters
        ----------
        ax : matplotlib.Axes or None, optional
            The axes on which to plot. If None plots on the currently active axes.
        relative_to : callable or None, optional
            If None it will plot the absolute timings, otherwise it will use the
            given *relative_to* function as reference for the timings.

        Raises
        ------
        ImportError
            If matplotlib isn't installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('simple_benchmark requires matplotlib for the '
                              'plotting functionality.')
        if ax is None:
            ax = plt.gca()

        x_axis = list(self._arguments)

        for func, timing in self._timings.items():
            label = self._function_name(func)
            if relative_to is None:
                plot_time = timing
            else:
                plot_time = [time / ref for time, ref in zip(self._timings[func], self._timings[relative_to])]
            ax.plot(x_axis, plot_time, label=label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(self._argument_name)
        if relative_to is None:
            ax.set_ylabel('time [seconds]')
        else:
            ax.set_ylabel('time relative to "{}"'.format(self._function_name(relative_to)))
        ax.grid(which='both')
        ax.legend()
        plt.tight_layout()

    def plot_both(self, relative_to):
        """Plot both the absolute times and the relative time.

        Parameters
        ----------
        relative_to : callable or None
            If None it will plot the absolute timings, otherwise it will use the
            given *relative_to* function as reference for the timings.

        Raises
        ------
        ImportError
            If matplotlib isn't installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('simple_benchmark requires matplotlib for the '
                              'plotting functionality')

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.plot(ax=ax1)
        self.plot(ax=ax2, relative_to=relative_to)
