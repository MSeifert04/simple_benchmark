Welcome to simple_benchmark's documentation!
============================================

Installation
------------

Using ``pip``:

.. code::

   python -m pip install simple_benchmark

Or installing the most recent version directly from ``git``:

.. code::

   python -m pip install git+https://github.com/MSeifert04/simple_benchmark.git

To utilize the all features of the library (for example visualization) you need to
install the optional dependencies:

- `NumPy <http://www.numpy.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `matplotlib <https://matplotlib.org/>`_

Getting started
---------------

Suppose you want to compare how NumPys sum and Pythons sum perform on lists
of different sizes::

    >>> from simple_benchmark import benchmark
    >>> import numpy as np
    >>> funcs = [sum, np.sum]
    >>> arguments = {i: [1]*i for i in [1, 10, 100, 1000, 10000, 100000]}
    >>> argument_name = 'list size'
    >>> aliases = {sum: 'Python sum', np.sum: 'NumPy sum'}
    >>> b = benchmark(funcs, arguments, argument_name, function_aliases=aliases)

The result can be visualized with ``pandas`` (needs to be installed)::

    >>> b
              Python sum  NumPy sum
    1       9.640884e-08   0.000004
    10      1.726930e-07   0.000004
    100     7.935484e-07   0.000008
    1000    7.040000e-06   0.000042
    10000   6.910000e-05   0.000378
    100000  6.899000e-04   0.003941

Or with ``matplotlib`` (has to be installed too)::

    >>> b.plot()

    >>> # To save the plotted benchmark as PNG file.
    >>> import matplotlib.pyplot as plt
    >>> plt.savefig('sum_example.png')

.. image:: ./sum_example.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   extended
   api
   changes
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
