# Licensed under Apache License Version 2.0 - see LICENSE
import argparse
import datetime
import importlib
import importlib.util
import inspect
import pathlib

import matplotlib.pyplot as plt

from simple_benchmark import BenchmarkBuilder


def _startswith_and_remainder(string, prefix):
    """Returns if the string starts with the prefix and the string without the prefix."""
    if string.startswith(prefix):
        return True, string[len(prefix):]
    else:
        return False, ''

def _import_file(filename, filepath):
    """This loads a python module by filepath"""
    spec = importlib.util.spec_from_file_location(filename, filepath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def _get_version_for_module(module_name):
    """Imports the module by its name and tries to get the version. Could fail..."""
    module = importlib.import_module(module_name)
    return module.__version__


def _hacky_parse_sig(function):
    """Extracts a useable alias by inspecting the function signature."""
    sig = inspect.signature(function)
    # Yeah, this looks for a parameter
    function_parameter = sig.parameters.get('func', None)
    if function_parameter:
        benchmarked_function = function_parameter.default
        if benchmarked_function:
            # __module__ will likely contain additional submodules. However
            # only the main module is (probably) of interest.
            module = benchmarked_function.__module__.split('.')[0]
            # Not every function has a __name__ attribute. But the rest of the
            # function is hacky too...
            name = benchmarked_function.__name__
            try:
                return f"{module} {name} ({_get_version_for_module(module)})"
            except Exception:
                # Something went wrong while determining the version. That's
                # okay, just omit it then...
                return f"{module} {name}"

def main(filename, outfilename, figsize, time_per_benchmark, write_csv, verbose):
    if verbose:
        print("Performing a Benchmark using simple_benchmark")
        print("---------------------------------------------")
        print("Effective Options:")
        print(f"Input-File: {filename}")
        print(f"Output-File: {outfilename}")
        print(f"Time per individual benchmark: {time_per_benchmark.total_seconds()} seconds")
        print(f"Figure size (inches): {figsize}")
        print(f"Verbose: {verbose}")

    path = pathlib.Path(filename).absolute()
    filename = path.name

    if verbose:
        print("")
        print("Process file")
        print("------------")

    module = _import_file(filename, path)

    b = BenchmarkBuilder(time_per_benchmark)

    for function_name in sorted(dir(module)):
        function = getattr(module, function_name)
        is_benchmark, benchmark_name = _startswith_and_remainder(function_name, 'bench_')
        if is_benchmark:
            try:
                alias = _hacky_parse_sig(function)
            except Exception:
                pass

            if not alias:
                alias = benchmark_name

            b.add_function(alias=alias)(function)
            continue

        is_args, args_name = _startswith_and_remainder(function_name, 'args_')
        if is_args:
            b.add_arguments(args_name)(function)
            continue

    if verbose:
        print("successful")
        print("")
        print("Running Benchmark")
        print("-----------------")
        print("this may take a while...")

    r = b.run()

    if verbose:
        print("successful")
        print("")
        print("Benchmark Result")
        print("----------------")
        print(r.to_pandas_dataframe())

    plt.figure(figsize=figsize)
    r.plot()
    plt.savefig(outfilename)

    out_file_path = pathlib.Path(outfilename)

    if verbose:
        print("")
        print(f"Written benchmark plot to {out_file_path.absolute()}")

    if write_csv:
        csv_file_path = out_file_path.with_suffix('.csv')
        # wtf ... pandas is using %-formatting ...
        # well, so %.9f should suppress scientific notation and display 9
        # decimals (nanosecond-resolution more is probably not useful anyway).
        r.to_pandas_dataframe().to_csv(str(csv_file_path), float_format='%.9f')
        print(f"Written CSV to {csv_file_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', help='the file to run the benchmark on.')
    parser.add_argument('out', help='Specifies the output file for the plot')
    parser.add_argument('-s', '--figsize', help='Specify the output size in inches, needs to be wrapped in quotes on most shells, e.g. "15, 9"', default='15, 9')
    parser.add_argument('--time-per-benchmark', help='The target time for each individual benchmark in seconds', default='0.1')
    parser.add_argument('-v', '--verbose', help='prints additional information on stdout', action="store_true")
    parser.add_argument('--write-csv', help='Writes an additional CSV file of the results', action="store_true")

    args = parser.parse_args()

    filename = args.filename
    outfilename = args.out

    verbose = args.verbose
    figsize = [int(value) for value in args.figsize.split(',')]
    time_per_benchmark = datetime.timedelta(seconds=float(args.time_per_benchmark))
    write_csv = args.write_csv
    main(filename, outfilename, figsize=figsize, time_per_benchmark=time_per_benchmark, write_csv=write_csv, verbose=verbose)
