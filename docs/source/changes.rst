Changelog
=========

0.0.8 (2019-04-06)
------------------

- Removed ``benchmark_random_list`` and ``benchmark_random_array`` in
  favor of the static methods ``use_random_lists_as_arguments`` and
  ``use_random_arrays_as_arguments`` on ``BenchmarkBuilder``.

- Added ``BenchmarkBuilder`` class that provides a decorator-based
  construction of a benchmark.

- Added a title to the plot created by the ``plot`` functions of
  ``BenchmarkResult`` that displays some information about the
  Python installation and environment.

0.0.7 (2018-04-30)
------------------

- Added optional ``estimator`` argument to the benchmark functions. The
  ``estimator`` can be used to calculate the reported runtime based on
  the individual timings.

0.0.6 (2018-04-30)
------------------

- Added ``plot_difference_percentage`` to ``BenchmarkResult`` to plot
  percentage differences.

0.0.5 (2018-04-22)
------------------

- Print a warning in case multiple functions have the same name

- Use ``OrderedDict`` to fix issues on older Python versions where ``dict``
  isn't ordered.

0.0.4 (2018-04-19)
------------------

- Added ``MultiArgument`` class to provide a way to pass in multiple
  arguments to the functions.

0.0.3 (2018-04-16)
------------------

- Some bugfixes.

0.0.2 (2018-04-16)
------------------

- General restructuring.

0.0.1 (2018-02-19)
------------------

- Initial release.