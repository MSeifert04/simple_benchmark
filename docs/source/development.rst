Development
===========

Prerequisites:

- Cloned or downloaded source repository. For example ``git clone https://github.com/MSeifert04/simple_benchmark.git``.
- You're in the root directory of the cloned (or downloaded) repository.
- Have an installed Python with pip and setuptools (the following will assume that the Python executable is in your path!).

Building the package locally
----------------------------

Navigate to the root directory of the repository (the directory where the ``setup.py`` file is) and then run one of
these commands::

   python setup.py develop

or::

   python -m pip install -e .

In case you want to install all the optional dependencies automatically (**recommended**)::

   python -m pip install -e .[optional]


Building the documentation locally
----------------------------------

This requires that the package was installed with all development dependencies::

   python -m pip install -e .[development]

Then just run::

   python setup.py build_sphinx

The generated HTML documentation should then be available in the
``./build/sphinx/html`` folder.


Running the tests locally
-------------------------

This requires that the package was installed with all development dependencies::

   python -m pip install -e .[development]

Then use ``pytest``::

    python -m pytest tests

Or to exclude the tests marked as slow::

    python -m pytest tests -m "not slow"


Publishing the package to PyPI
------------------------------

.. note::
   This is maintainer-only!

To install the necessary packages run::

   python -m pip install -e .[maintainer]

First clean the repository to avoid outdated artifacts::

   git clean -dfX

Then build the source distribution, since it's a very small package without compiled modules, we can omit building
wheels::

   python setup.py sdist

Then upload to PyPI::

   python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

You will be prompted for the username and password.
