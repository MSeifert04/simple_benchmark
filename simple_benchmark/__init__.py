# Licensed under Apache License Version 2.0 - see LICENSE
"""
.. warning::
   This package is under active development. API changes are very likely.

This package aims to give an easy way to benchmark several functions for
different inputs and provide ways to visualize the benchmark results.

To utilize the full features (visualization and post-processing) you need to
install the optional dependencies:

- NumPy
- pandas
- matplotlib
"""

__version__ = '0.1.0'

__all__ = [
    'assert_same_results', 'assert_not_mutating_input', 'benchmark',
    'BenchmarkBuilder', 'BenchmarkResult', 'MultiArgument'
]

import copy
import functools
import itertools
import platform
import pprint
import random
import sys
import timeit
import warnings

from collections import OrderedDict

_DEFAULT_ARGUMENT_NAME = ''
_DEFAULT_TIME_PER_BENCHMARK = 0.1
_DEFAULT_ESTIMATOR = min
_DEFAULT_COPY_FUNC = copy.deepcopy

_MISSING = object()
_MSG_DECORATOR_FACTORY = (
    'A decorator factory cannot be applied to a function directly. The decorator factory returns a decorator when '
    'called so if no arguments should be applied then simply call the decorator factory without arguments.'
)
_MSG_MISSING_ARGUMENTS = "The BenchmarkBuilder instance is missing arguments for the functions."


class MultiArgument(tuple):
    """Class that behaves like a tuple but signals to the benchmark that it
    should pass multiple arguments to the function to benchmark.
    """
    pass


def _try_importing_matplotlib():
    """Tries to import matplotlib

    Returns
    -------
    pyplot : module
        The pyplot module from matplotlib

    Raises
    ------
    ImportError
        In case matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('simple_benchmark requires matplotlib for the '
                          'plotting functionality.')
    return plt


def _get_python_bits():
    """Is the current platform 64bit.

    Returns
    -------
    result : string
        The string '64bit' in case the Python installation uses 64bit or more, otherwise '32bit'.
    """
    return '64bit' if sys.maxsize > 2 ** 32 else '32bit'


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
        return max(int(n_repeats // factor), 1), max(int(factor), 1)
    # Otherwise the number of timings each repeat should be 1.
    # However make sure there are at least 3 repeats for each function!
    return max(n_repeats, 3), 1


def _get_bound_func(func, argument):
    if isinstance(argument, MultiArgument):
        return functools.partial(func, *argument)
    else:
        return functools.partial(func, argument)


def assert_same_results(funcs, arguments, equality_func):
    """Asserts that all functions return the same result.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    funcs : iterable of callables
        The functions to check.
    arguments : dict
        A dictionary containing where the key represents the reported value
        (for example an integer representing the list size) as key and the argument
        for the functions (for example the list) as value.
        In case you want to plot the result it should be sorted and ordered
        (e.g. an :py:class:`collections.OrderedDict` or a plain dict if you are
        using Python 3.7 or later).
    equality_func : callable
        The function that determines if the results are equal. This function should
        accept two arguments and return a boolean (True if the results should be
        considered equal, False if not).

    Raises
    ------
    AssertionError
        In case any two results are not equal.
    """
    funcs = list(funcs)
    for arg in arguments.values():
        first_result = _MISSING
        for func in funcs:
            bound_func = _get_bound_func(func, arg)
            result = bound_func()
            if first_result is _MISSING:
                first_result = result
            else:
                assert equality_func(first_result, result), (func, first_result, result)


def assert_not_mutating_input(funcs, arguments, equality_func, copy_func=_DEFAULT_COPY_FUNC):
    """Asserts that none of the functions mutate the arguments.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    funcs : iterable of callables
        The functions to check.
    arguments : dict
        A dictionary containing where the key represents the reported value
        (for example an integer representing the list size) as key and the argument
        for the functions (for example the list) as value.
        In case you want to plot the result it should be sorted and ordered
        (e.g. an :py:class:`collections.OrderedDict` or a plain dict if you are
        using Python 3.7 or later).
    equality_func : callable
        The function that determines if the results are equal. This function should
        accept two arguments and return a boolean (True if the results should be
        considered equal, False if not).
    copy_func : callable, optional
        The function that is used to copy the original argument.
        Default is :py:func:`copy.deepcopy`.

    Raises
    ------
    AssertionError
        In case any two results are not equal.

    Notes
    -----
    In case the arguments are :py:class:`MultiArgument` then the copy_func and the
    equality_func get these :py:class:`MultiArgument` as single arguments and need
    to handle them appropriately.
    """
    funcs = list(funcs)
    for arg in arguments.values():
        original_arguments = copy_func(arg)
        for func in funcs:
            bound_func = _get_bound_func(func, arg)
            bound_func()
            assert equality_func(original_arguments, arg), (func, original_arguments, arg)


def benchmark(
        funcs,
        arguments,
        argument_name=_DEFAULT_ARGUMENT_NAME,
        warmups=None,
        time_per_benchmark=_DEFAULT_TIME_PER_BENCHMARK,
        function_aliases=None,
        estimator=_DEFAULT_ESTIMATOR):
    """Create a benchmark suite for different functions and for different arguments.

    Parameters
    ----------
    funcs : iterable of callables
        The functions to benchmark.
    arguments : dict
        A dictionary containing where the key represents the reported value
        (for example an integer representing the list size) as key and the argument
        for the functions (for example the list) as value.
        In case you want to plot the result it should be sorted and ordered
        (e.g. an :py:class:`collections.OrderedDict` or a plain dict if you are
        using Python 3.7 or later).
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
    estimator : callable, optional
        Each function is called with each argument multiple times and each
        timing is recorded. The benchmark_estimator (by default :py:func:`min`)
        is used to reduce this list of timings to one final value.
        The minimum is generally a good way to estimate how fast a function can
        run (see also the discussion in :py:meth:`timeit.Timer.repeat`).
        Default is :py:func:`min`.

    Returns
    -------
    benchmark : BenchmarkResult
        The result of the benchmarks.

    See also
    --------
    BenchmarkBuilder
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
            bound_func = _get_bound_func(func, arg)
            for _ in itertools.repeat(None, times=warm_up_calls[func]):
                bound_func()
            repeats, number = _estimate_number_of_repeats(bound_func, time_per_benchmark)
            # As per the timeit module documentation a very good approximation
            # of a timing is found by repeating the benchmark and using the
            # minimum.
            times = timeit.repeat(bound_func, number=number, repeat=repeats)
            time = estimator(times)
            timing_list.append(time / number)
    return BenchmarkResult(timings, function_aliases, arguments, argument_name)


class BenchmarkResult(object):
    """A class holding a benchmarking result that provides additional printing and plotting functions."""
    def __init__(self, timings, function_aliases, arguments, argument_name):
        self._timings = timings
        self.function_aliases = function_aliases
        self._arguments = arguments
        self._argument_name = argument_name

    def __str__(self):
        """Prints the results as table."""
        try:
            return str(self.to_pandas_dataframe())
        except ImportError:
            return pprint.pformat({self._function_name(k): v for k, v in self._timings.items()})

    __repr__ = __str__

    def _function_name(self, func):
        """Returns the associated name of a function."""
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

    @staticmethod
    def _get_title():
        """Returns a string containing some information about Python and the machine."""
        return "{0} {1} {2} ({3} {4})".format(
            platform.python_implementation(), platform.python_version(), platform.python_compiler(),
            platform.system(), platform.release())

    def to_pandas_dataframe(self):
        """Return the timing results as pandas DataFrame. This is the preferred way of accessing the text form of the
        timings.

        Returns
        -------
        results : pandas.DataFrame
            The timings as DataFrame.

        Warns
        -----
        UserWarning
            In case multiple functions have the same name.

        Raises
        ------
        ImportError
            If pandas isn't installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('simple_benchmark requires pandas for this method.')
        dct = {self._function_name(func): timings for func, timings in self._timings.items()}
        if len(dct) != len(self._timings):
            warnings.warn('Some timings are not included in the result. Likely '
                          'because multiple functions have the same name. You '
                          'can add an alias to the `function_aliases` mapping '
                          'to avoid this problem.', UserWarning)

        return pd.DataFrame(dct, index=list(self._arguments))

    def plot_difference_percentage(self, relative_to, ax=None):
        """Plot the benchmarks relative to one of the benchmarks with
        percentages on the y-axis.

        Parameters
        ----------
        relative_to : callable
            The benchmarks are plotted relative to the timings of the given
            function.
        ax : matplotlib.axes.Axes or None, optional
            The axes on which to plot. If None plots on the currently active axes.

        Raises
        ------
        ImportError
            If matplotlib isn't installed.
        """
        plt = _try_importing_matplotlib()
        ax = ax or plt.gca()

        self.plot(relative_to=relative_to, ax=ax)

        ax.set_yscale('linear')
        # Use percentage always including the sign for the y ticks.
        ticks = ax.get_yticks()
        ax.set_yticklabels(['{:+.1f}%'.format((x-1) * 100) for x in ticks])

    def plot(self, relative_to=None, ax=None):
        """Plot the benchmarks, either relative or absolute.

        Parameters
        ----------
        relative_to : callable or None, optional
            If None it will plot the absolute timings, otherwise it will use the
            given *relative_to* function as reference for the timings.
        ax : matplotlib.axes.Axes or None, optional
            The axes on which to plot. If None plots on the currently active axes.

        Raises
        ------
        ImportError
            If matplotlib isn't installed.
        """
        plt = _try_importing_matplotlib()
        ax = ax or plt.gca()

        x_axis = list(self._arguments)

        for func, timing in self._timings.items():
            label = self._function_name(func)
            if relative_to is None:
                plot_time = timing
            else:
                plot_time = [time / ref for time, ref in zip(self._timings[func], self._timings[relative_to])]
            ax.plot(x_axis, plot_time, label=label)

        ax.set_title(BenchmarkResult._get_title())
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(self._argument_name)
        if relative_to is None:
            ax.set_ylabel('time [seconds]')
        else:
            ax.set_ylabel('time relative to "{}"'.format(self._function_name(relative_to)))
        ax.grid(which='both')
        ax.legend()

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
        plt = _try_importing_matplotlib()

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.plot(ax=ax1)
        self.plot(ax=ax2, relative_to=relative_to)


class BenchmarkBuilder(object):
    """A class useful for building benchmarks by adding decorators to the functions instead of collecting them later.

    Parameters
    ----------
    time_per_benchmark : float, optional
        Each benchmark should take approximately this value in seconds.
        However the value is ignored for functions that take very little time
        or very long.
        Default is 0.1 (seconds).
    estimator : callable, optional
        Each function is called with each argument multiple times and each
        timing is recorded. The benchmark_estimator (by default :py:func:`min`)
        is used to reduce this list of timings to one final value.
        The minimum is generally a good way to estimate how fast a function can
        run (see also the discussion in :py:meth:`timeit.Timer.repeat`).
        Default is :py:func:`min`.

    See also
    --------
    benchmark
    """
    def __init__(self, time_per_benchmark=_DEFAULT_TIME_PER_BENCHMARK, estimator=_DEFAULT_ESTIMATOR):
        self._funcs = []
        self._arguments = OrderedDict()
        self._warmups = []
        self._function_aliases = {}
        self._argument_name = _DEFAULT_ARGUMENT_NAME
        self._time_per_benchmark = time_per_benchmark
        self._estimator = estimator

    def add_functions(self, functions):
        """Add multiple functions to the benchmark.

        Parameters
        ----------
        functions : iterable of callables
            The functions to add to the benchmark
        """
        self._funcs.extend(functions)

    def add_function(self, warmups=False, alias=None):
        """A decorator factory that returns a decorator that can be used to add a function to the benchmark.

        Parameters
        ----------
        warmups : bool, optional
            If true the function is called once before each benchmark run.
            Default is False.
        alias : str or None, optional
            If None then the displayed function name is the name of the function, otherwise the string is used when
            the function is referred to.
            Default is None.

        Returns
        -------
        decorator : callable
            The decorator that adds the function to the benchmark.

        Raises
        ------
        TypeError
            In case ``name`` is a callable.
        """
        if callable(warmups):
            raise TypeError(_MSG_DECORATOR_FACTORY)

        def inner(func):
            self._funcs.append(func)
            if warmups:
                self._warmups.append(func)
            if alias is not None:
                self._function_aliases[func] = alias
            return func

        return inner

    def add_arguments(self, name=_DEFAULT_ARGUMENT_NAME):
        """A decorator factory that returns a decorator that can be used to add a function that produces the x-axis
         values and the associated test data for the benchmark.

        Parameters
        ----------
        name : str, optional
            The label for the x-axis.

        Returns
        -------
        decorator : callable
            The decorator that adds the function that produces the x-axis values and the test data to the benchmark.

        Raises
        ------
        TypeError
            In case ``name`` is a callable.
        """
        if callable(name):
            raise TypeError(_MSG_DECORATOR_FACTORY)

        def inner(func):
            self._arguments = OrderedDict(func())
            self._argument_name = name
            return func

        return inner

    def assert_same_results(self, equality_func):
        """Asserts that all stored functions return the same result.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        equality_func : callable
            The function that determines if the results are equal. This function should
            accept two arguments and return a boolean (True if the results should be
            considered equal, False if not).

        Warns
        -----
        UserWarning
            In case the instance has no arguments for the functions.

        Raises
        ------
        AssertionError
            In case any two results are not equal.
        """
        if not self._arguments:
            warnings.warn(_MSG_MISSING_ARGUMENTS, UserWarning)
            return
        assert_same_results(self._funcs, self._arguments, equality_func=equality_func)

    def assert_not_mutating_input(self, equality_func, copy_func=_DEFAULT_COPY_FUNC):
        """Asserts that none of the stored functions mutate the arguments.

        .. versionadded:: 0.1.0

        Parameters
        ----------
        equality_func : callable
            The function that determines if the results are equal. This function should
            accept two arguments and return a boolean (True if the results should be
            considered equal, False if not).
        copy_func : callable, optional
            The function that is used to copy the original argument.
            Default is :py:func:`copy.deepcopy`.

        Warns
        -----
        UserWarning
            In case the instance has no arguments for the functions.

        Raises
        ------
        AssertionError
            In case any two results are not equal.

        Notes
        -----
        In case the arguments are :py:class:`MultiArgument` then the copy_func and the
        equality_func get these :py:class:`MultiArgument` as single arguments and need
        to handle them appropriately.
        """
        if not self._arguments:
            warnings.warn(_MSG_MISSING_ARGUMENTS, UserWarning)
            return
        assert_not_mutating_input(self._funcs, self._arguments, equality_func=equality_func, copy_func=copy_func)

    def run(self):
        """Starts the benchmark.

        Returns
        -------
        result : BenchmarkResult
            The result of the benchmark.

        Warns
        -----
        UserWarning
            In case the instance has no arguments for the functions.
        """
        if not self._arguments:
            warnings.warn(_MSG_MISSING_ARGUMENTS, UserWarning)
        return benchmark(
            funcs=self._funcs,
            arguments=self._arguments,
            argument_name=self._argument_name,
            warmups=self._warmups,
            time_per_benchmark=self._time_per_benchmark,
            function_aliases=self._function_aliases,
            estimator=self._estimator)

    def use_random_arrays_as_arguments(self, sizes):
        """Alternative to :meth:`add_arguments` that provides random arrays of the specified sizes as arguments for the
        benchmark.

        Parameters
        ----------
        sizes : iterable of int
            An iterable containing the sizes for the arrays (should be sorted).

        Raises
        ------
        ImportError
            If NumPy isn't installed.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError('simple_benchmark requires NumPy for this function.')

        def provide_random_arrays():
            for size in sizes:
                yield size, np.random.random(size)

        self.add_arguments('array size')(provide_random_arrays)

    def use_random_lists_as_arguments(self, sizes):
        """Alternative to :meth:`add_arguments` that provides random lists of the specified sizes as arguments for the
        benchmark.

        Parameters
        ----------
        sizes : iterable of int
            An iterable containing the sizes for the lists (should be sorted).
        """
        random_func = random.random

        def provide_random_lists():
            for size in sizes:
                yield size, [random_func() for _ in itertools.repeat(None, times=size)]

        self.add_arguments('list size')(provide_random_lists)
