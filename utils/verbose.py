#!/usr/bin/env python

"""
Introduction
============
Define useful helper functions.
"""
import time
import logging
import resource
import argparse
import subprocess
import functools
import tracemalloc
from time import sleep
from typing import Dict
from os.path import exists
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import psutil


def logging_time(original_fn=None, verbose=False):
    """Logging Elapsed Time.

    Note:
        wrapped function's argument should not have `report`.

    Examples:
        >>> # Basically, decorator logging_time results in as follows.
        >>> @logging_time
        >>> def func(...)
        >>> result, elapsed_time = func(...)
        >>> # If you want to print elapsed time verbosely.
        >>> @logging_time(verbose=True)
        >>> def func(...)
        >>> # func(...) result in as follows.
        >>> # ElapsedTime[func]: [elapsed time] [ms]
        >>> # If you does not want to logging elapsed time explicitly with slience.
        >>> result = func(..., sclience=True)
    """
    def _logging_time(original_fn):
        """Decorator which takes a role in logging elapsed time for a given function. """
        def wrapper(*args, **kwargs):
            slience = False
            if 'slience' in kwargs:
                slience = kwargs['slience']
                del kwargs['slience']
            nonlocal verbose
            if 'verbose' in kwargs:
                verbose = kwargs['verbose']
                del kwargs['verbose']
            start_time = time.time()
            result = original_fn(*args, **kwargs)
            elapsed_time = (time.time() - start_time) * 1e3
            unit = 'ms'
            if elapsed_time > 1000:
                elapsed_time /= 1000
                unit = 'sec'
            if slience:
                return result
            elif verbose:
                print(f"ElapsedTime[{original_fn.__name__}]: {elapsed_time:.4f} [{unit}]")
                return result
            else:
                return result, elapsed_time
        return wrapper
    if original_fn:
        return _logging_time(original_fn)
    return _logging_time


def logging_sampling_peak_memory(original_fn=None, sampling_period=0.1, verbose=True):
    """Decorator which takes a role in sampling peak memory usage for a given function. """
    def _logging_sampling_peak_memory(original_fn):
        def wrapper(*args, **kwargs):
            nonlocal verbose
            if 'verbose' in kwargs:
                verbose = kwargs['verbose']
                del kwargs['verbose']
            if verbose:
                print(f"available physical memory: {psutil.virtual_memory().available / (1024 ** 3):.4f} [GB]")
                return sampling_peak_memory(
                    original_fn,
                    *args,
                    sampling_period=sampling_period,
                    verbose=verbose,
                    **kwargs)
            else:
                return original_fn(*args, **kwargs)
        return wrapper
    if original_fn:
        return _logging_sampling_peak_memory(original_fn)
    return _logging_sampling_peak_memory


def sampling_peak_memory(original_fn, *args, sampling_period=0.1, **kwargs):
    if not exists('/etc/os-release'):
        print("Get peak memory for only linux system.")
        return
    result = None
    with ThreadPoolExecutor() as executor:
        monitor = MemoryMonitor(sampling_period)
        mem_thread = executor.submit(monitor.measure_usage)
        try:
            fn_thread = executor.submit(original_fn, *args, **kwargs)
            result = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            max_usage = mem_thread.result()
            unit = 'KB'
            if 10 ** 3 < max_usage < 10 ** 6:
                unit = 'MB'
                max_usage /= float(10**3)
            elif 10 ** 6 < max_usage:
                unit = 'GB'
                max_usage /= float(10 ** 6)
        print(f"Peak memory usage: {max_usage:.4f} [{unit}]")
    return result


def get_logging_format():
    fmt = '[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s'
    return fmt


def get_logging_datefmt():
    return '%Y-%m-%d %H:%M:%S'


def get_num_lines(fname):
    num_lines = int(subprocess.getoutput('wc -l ' + fname + '| awk \'{print $1}\''))
    return num_lines + 1


def printProgressBar(iteration, total, msg, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """Call in a loop to create terminal progress bar

    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    show = '\r{}|{}| {} {}% - {}'.format(prefix, bar, percent, suffix, msg)
    print(show, end='\r')


class MemoryMonitor:
    """Memory moniter class.

    Note:
        Please see this article.
            https://medium.com/survata-engineering-blog/monitoring-memory-usage-of-a-running-python-program-49f027e3d1ba
    """
    def __init__(self, sampling_period=0.1):
        self.keep_measuring = True
        self.sampling_period = sampling_period

    def measure_usage(self):
        max_usage = 0
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            sleep(self.sampling_period)
        return max_usage
