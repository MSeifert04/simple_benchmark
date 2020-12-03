Command Line
============

Using the Command Line
----------------------

.. warning::
   The command line interface is highly experimental. It's very likely to
   change its API.

When you have all optional dependencies installed you can also run
``simple_benchmark``, in the most basic form it would be::

    $ python -m simple_benchmark INPUT_FILE OUTPUT_FILE

Which processes the ``INPUT_FILE`` and writes a plot to ``OUTPUT_FILE``.

However in order to work correctly the ``INPUT_FILE`` has to fulfill several
criteria:

- It must be a valid Python file.
- All functions that should be benchmarked have to have a name starting with ``bench_``
  and everything thereafter is used for the label.
- The function generating the arguments for the benchmark has to start with ``args_``
  and everything thereafter is used for the label of the x-axis.

Also if the benchmarked function has a ``func`` parameter with a default it
will be used to determine the ``alias`` (the displayed name in the table and
plot).


Parameters
----------

The first two parameters are the input and output file. However there are a
few more parameters. These can be also seen when running::

    $ python -m simple_benchmark -h
    usage: __main__.py [-h] [-s FIGSIZE] [--time-per-benchmark TIME_PER_BENCHMARK] [-v] [--write-csv] filename out

    Benchmark a file

    positional arguments:
    filename              the file to run the benchmark on.
    out                   Specifies the output file for the plot

    optional arguments:
    -h, --help            show this help message and exit
    -s FIGSIZE, --figsize FIGSIZE
                            Specify the output size in inches, needs to be wrapped in quotes on most shells, e.g. "15, 9" (default: 15, 9)
    --time-per-benchmark TIME_PER_BENCHMARK
                            The target time for each individual benchmark in seconds (default: 0.1)
    -v, --verbose         prints additional information on stdout (default: False)
    --write-csv           Writes an additional CSV file of the results (default: False)
