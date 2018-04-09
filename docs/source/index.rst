.. simple_benchmark documentation master file, created by
   sphinx-quickstart on Fri Mar  9 16:14:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to simple_benchmark's documentation!
============================================

Installation
------------

Using ``pip``:

.. code::

   pip install simple_benchmark

Or the manual installation using ``git``:

.. code::

   git clone https://github.com/MSeifert04/simple_benchmark.git
   cd simple_benchmark
   python setup.py install

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
            NumPy sum    Python sum
    1        0.000003  1.032715e-07
    10       0.000004  1.569619e-07
    100      0.000007  7.155641e-07
    1000     0.000042  6.153851e-06
    10000    0.000382  6.030774e-05
    100000   0.004034  6.026672e-04

Or with ``matplotlib`` (has to be installed too)::

    >>> b.plot()

.. image:: ./sum_example.png

.. automodule:: simple_benchmark
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
