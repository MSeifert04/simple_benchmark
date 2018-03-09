# Licensed under Apache License Version 2.0 - see LICENSE

"""A simple benchmarking package."""

__version__ = '0.0.1'

__all__ = ['Benchmark']

import functools
import itertools
import pprint
import random
import timeit


def _estimate_number_of_repeats(func, target_seconds):
    """Estimate the number of repeats for a function
    so that the benchmark will take a specific time.

    In case the function is much slower or really fast
    some default values are returned.

    Parameters
    ----------
    func : callable
        The function to time. Must not have required arguments!
    target_seconds : number
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

    # Determine the number of repeats and split it into
    # two numbers that represent the repeat and number
    # argument for timeit.repeat.
    n_repeats = int(target_seconds / single_time)
    if n_repeats < 3:
        return 3, 1
    elif n_repeats > 100000:
        return 10000, 15
    elif n_repeats > 1000:
        return n_repeats // 7, 7
    else:
        return n_repeats, 1


class Benchmark(object):
    """Create a benchmark suite for different functions and for different arguments.

    Parameters
    ----------
    funcs : iterable of callables
        The functions to benchmark.
    arguments : dictionary
        A dictionary containing the "metric value" as key and the argument
        for the function as value.
    argument_name : str
        The name of the argument. For example if the arguments are different
        sizes this could be "size".
    warmups : None or iterable of callables, optional
        If not None it specifies the callables that need a warmup call
        before being timed. That is so, that caches can be filled or
        jitters to kick in.
        Default is None.
    time_per_benchmark : number, optional
        Each benchmark should take approximately this value in seconds.
        However the value is ignored for functions that take very little time
        or very long.
        Default is 0.1 (seconds).
    function_aliases : None or dict, optional
        If not None it should be a dictionary containing the function as key
        and the name of the function as value. The value will be used in the
        final plots.
        Default is None.

    Examples
    --------
    For example to benchmark different sum functions on a Python list.

    The setup::

        >>> from simple_benchmark import Benchmark
        >>> import numpy as np
        >>> funcs = [sum, np.sum]
        >>> arguments = {1: [10],
        ...              5: [2, 5, 10, 20, 40],
        ...              100: [2]*100}
        >>> argument_name = 'list size'
        >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
        >>> b = Benchmark(funcs, arguments, argument_name, function_aliases=aliases)

    Running the benchmarks::

        >>> b.run()

    Inspecting the results::

        >>> b.to_pandas_dataframe()

    Plotting the results::

        >>> b.plot()
        >>> b.plot(relative_to=np.sum)
        >>> b.plot_both(sum)

    In case the arguments are NumPy arrays or lists containing random floats
    there are easier ways with different constructors::

        >>> from simple_benchmark import Benchmark
        >>> import numpy as np
        >>> funcs = [sum, np.sum]
        >>> sizes = [i * 10 for i in range(20)]
        >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
        >>> b = Benchmark.from_random_array_sizes(funcs, sizes, function_aliases=aliases)
        >>> b.run()
    """

    def __init__(
            self,
            funcs,
            arguments,
            argument_name,
            warmups=None,
            time_per_benchmark=0.1,
            function_aliases=None):
        self._timings = {func: [] for func in funcs}
        self._warmup = {func: 0 for func in funcs}
        if warmups is not None:
            for func in warmups:
                self._warmup[func] = 1
        self._arguments = {value: arg for value, arg in arguments.items()}
        self._argument_name = argument_name
        self._time = time_per_benchmark
        self._function_aliases = function_aliases or {}
        self._ran = False

    @classmethod
    def from_random_array_sizes(
            cls,
            funcs,
            sizes,
            warmups=None,
            time_per_benchmark=0.1,
            function_aliases=None):
        """A shortcut constructor if a random array is wanted.

        The arguments *arguments* and *argument_name* of the normal constructor
        are replaced with a simple *size* argument.

        Parameters
        ----------
        sizes : iterable of integers
            The different size values for arrays.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError('simple_benchmark requires NumPy for this function.')
        return cls(funcs,
                   arguments={size: np.random.random(size) for size in sizes},
                   argument_name='array size',
                   warmups=warmups,
                   time_per_benchmark=time_per_benchmark,
                   function_aliases=function_aliases)

    @classmethod
    def from_random_list_sizes(
            cls,
            funcs,
            sizes,
            warmups=None,
            time_per_benchmark=0.1,
            function_aliases=None):
        """A shortcut constructor if a random list is wanted.

        The arguments *arguments* and *argument_name* of the normal constructor
        are replaced with a simple *size* argument.

        Parameters
        ----------
        sizes : iterable of integers
            The different size values for list.
        """
        random_func = random.random
        return cls(funcs,
                   arguments={size: [random_func() for _ in itertools.repeat(None, times=size)]
                              for size in sizes},
                   argument_name='list size',
                   warmups=warmups,
                   time_per_benchmark=time_per_benchmark,
                   function_aliases=function_aliases)

    def __str__(self):
        if self._ran:
            return pprint.pformat(self._timings)
        else:
            return '<{} (not run yet)>'.format(type(self).__name__)

    def _function_name(self, func):
        try:
            return self._function_aliases[func]
        except KeyError:
            # Has to be a different branch because not every function has a
            # __name__ attribute. So we cannot simply use the dictionaries `get`
            # with default.
            try:
                return func.__name__
            except AttributeError:
                raise TypeError('function "func" does not have a __name__ attribute. '
                                'Please use "function_aliases" to provide a function name alias.')

    def run(self):
        """Run the benchmarks."""
        for arg in self._arguments.values():
            for func, timing_list in self._timings.items():
                bound_func = functools.partial(func, arg)
                for _ in itertools.repeat(None, times=self._warmup[func]):
                    bound_func()
                repeats, number = _estimate_number_of_repeats(bound_func, self._time)
                times = timeit.repeat(bound_func, number=number, repeat=repeats)
                time = min(times)
                timing_list.append(time / number)
        self._ran = True

    def to_pandas_dataframe(self):
        """Return the timing results as pandas Dataframe. This is the preferred way of accessing the timings.

        Requires Pandas.

        Returns
        -------
        df : pandas.DataFrame
            The timings as DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('simple_benchmark requires pandas for this method.')
        if not self._ran:
            raise ValueError('You have to run the benchmarks before you can convert them.')
        return pd.DataFrame(
            {self._function_name(func): timings for func, timings in self._timings.items()},
            index=list(self._arguments))

    def plot(self, ax=None, relative_to=None):
        """Plot the benchmarks, either relative or absolute.

        Requires matplotlib.

        Parameters
        ----------
        ax : matplotlib.Axes or None, optional
            The axes on which to plot. If None plots on the currently active axes.
        relative_to : callable or None, optional
            If None it will plot the absolute timings, otherwise it will use the
            given *relative_to* function as reference for the timings.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('simple_benchmark requires Matplotlib for the '
                              'plotting functionality.')
        if not self._ran:
            raise ValueError('You have to run the benchmarks before you can plot them.')
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

        Requires matplotlib.

        Parameters
        ----------
        relative_to : callable or None
            If None it will plot the absolute timings, otherwise it will use the
            given *relative_to* function as reference for the timings.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('simple_benchmark requires Matplotlib for the '
                              'plotting functionality')

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.plot(ax=ax1)
        self.plot(ax=ax2, relative_to=relative_to)
