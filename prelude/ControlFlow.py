#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""


def retry(call, tries, errors=Exception):
    for attempt in range(tries):
        try:
            return call()
        except errors:
            if attempt + 1 == tries:
                raise


def ignore(call, errors=Exception):
    try:
        return call()
    except errors:
        return None



def in_sequence(function_list):
    """
    Create a new function that execute the functions
    in the list in sequence.

    :param function_list: List of functions
    :return:              Function

    """

    def seqfun():
        for f in function_list: f()

    return seqfun

import threading

def in_parallel(function_list):
    """
    Create a new function that execute the functions
    in the list in parallel (thread)

    :param function_list: List of functions
    :return:              Function

    Example:

    >>> from m2py import functional as funcp
    >>>
    >>> import time
    >>>
    >>> def print_time(thname, delay):
    ...   for i in range(5):
    ...      time.sleep(delay)
    ...      print (thname, " ", time.ctime(time.time()))
    >>>
    >>> def make_print_time(name, delay): return lambda : print_time(name, delay)
    >>>
    >>> t1 = make_print_time("thread1", 1)
    >>> t2 = make_print_time("thread2", 2)
    >>> t3 = make_print_time("thread3", 3)
    >>> t4 = make_print_time("thread4", 4)
    >>>
    >>> thfun = funcp.in_parallel([t1, t2, t3, t4])
    >>> thfun()
    >>> thread1   Fri Dec 26 23:40:29 2014
    thread2   Fri Dec 26 23:40:30 2014
    thread1   Fri Dec 26 23:40:30 2014
    thread3   Fri Dec 26 23:40:31 2014
    thread1   Fri Dec 26 23:40:31 2014
    thread4   Fri Dec 26 23:40:32 2014
    thread2   Fri Dec 26 23:40:32 2014
    thread1   Fri Dec 26 23:40:32 2014
    thread1   Fri Dec 26 23:40:33 2014
    ...
    """
    nThtreads = len(function_list)

    results = [None] * nThtreads

    def adapter(func, idx):
        results[idx] = func()


    for idx, func in enumerate(function_list):
        t = threading.Thread(
            target=adapter,
            args = (func, idx)
        )
        t.daemon = True
        t.start()
        t.join()
        #thread.start_new_thread(adapter, (func, idx))


    return results




def ifelseDo(predicate, fa, fb):
    """

    ifelseDo :: (a -> bool) -> (a -> b) -> (a -> b) -> (a -> b)

        predicate :: a -> bool
        fa        :: a -> b
        fb        :: a -> b

    f(x) = ifelse(predicate, fa, fb)


    Example:

    Crete
                /  x^2  , if x < 3
        f(x)  =
                \  x/3  , if x >= 3

    """

    def _(x):
        return [fa, fb][predicate(x)](x)
    return _


def ifelse(predicate, a, b):
    """
    ifelse :: (s -> bool) -> a -> b -> ( s -> OR(a, b))
    """
    return lambda x: [a, b][predicate(x)]


