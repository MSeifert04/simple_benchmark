import sys

from setuptools import setup, find_packages


package_name = "simple_benchmark"

optional_dependencies = ["numpy", "matplotlib", "pandas"]
development_dependencies = ["sphinx", "pytest"]
maintainer_dependencies = ["twine"]


def readme():
    with open('README.rst') as f:
        return f.read()


def version():
    with open('{}/__init__.py'.format(package_name)) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split(r"'")[1]


setup(name=package_name,
      version=version(),

      description='A simple benchmarking package.',
      long_description=readme(),
      # Somehow the keywords get lost if I use a list of strings so this is
      # just a longish string...
      keywords='performance timing timeit',
      platforms=["Windows Linux Mac OS-X"],

      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Utilities',
        'Topic :: System :: Benchmark'
      ],

      license='Apache License Version 2.0',

      url='https://github.com/MSeifert04/simple_benchmark',

      author='Michael Seifert',
      author_email='michaelseifert04@yahoo.de',

      packages=find_packages(exclude=['ez_setup']),

      tests_require=["pytest"],
      extras_require={
          'optional': optional_dependencies,
          'development': optional_dependencies + development_dependencies,
          'maintainer': optional_dependencies + development_dependencies + maintainer_dependencies
      },

      include_package_data=True,
      zip_safe=False,
)
