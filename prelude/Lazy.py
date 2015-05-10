#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colelctions of functions to manipulate Streams/ Generators for Lazy
evaluation.

"""
import itertools


from .prelude import curry

def nthst(stream, n):
    """

    nth(N, List) -> Elem

    > lists:nth(3, [a, b, c, d, e]).
    c

    Idea from: http://erldocs.com/17.3/stdlib/lists.html
    """

    i = 0
    while i < n:
        next(stream)  # Consume stream
        i += 1
    return next(stream)


def zipWith(func, stream1, stream2):
    return map(lambda e: func(e[0], e[1]), zip(stream1, stream2))

@curry
def take(n, iterable):
    for i in range(n):

        try:
            yield next(iterable)
        except:
            pass

@curry
def takel(n, iterable):

    result = []

    for i in range(n):

        try:
            result.append(next(iterable))
        except:
            pass

    return result

@curry
def takeWhile(predicate, stream):
    while True:

        x = next(stream)

        if not predicate(x):
            break

        try:
            yield x
        except StopIteration:
            break

@curry
def takeWhileNext(predicate, stream):
    """
    Returns one more iteration after reach the predicate

    """
    while True:

        x = next(stream)
        #print("x = ", x)

        if not predicate(x):
            #print("Last yield")
            #print("x = ", x)
            yield x
            break

        try:
            yield x
        except StopIteration:
            break


@curry
def dropWhile(predicate, stream):
    while True:

        x = next(stream)

        if not predicate(x):
            try:
                yield x
            except StopIteration:
                break


def flat(stream):
    return itertools.chain(*stream)


@curry
def flat_mapl(function, stream):
    return flat(map(function, stream))

@curry
def iterate(f, x):
    y = x

    while True:

        try:
            yield y
            y = f(y)
        except StopIteration:
            break


def tail(stream):
    next(stream)
    return stream


def lastl(alist):
    return alist[-1]

def last(stream):
    """
    Return last element of generator
    and discard the remaining.
    """

    current = None

    while True:

        try:
            current = next(stream)
        except StopIteration:
            return current



def head(stream):
    return next(stream)


def drop(n, stream):
    for i in range(n):
        next(stream)

    return stream

@curry
def foreach(function, iterable):
    for element in iterable:
        function(next(iterable))

@curry
def mapf(function, iterable):
    return map(function, iterable)

def pairs(alist):
    return zip(alist, tail(iter(alist)))


def pairsl(alist):
    return list(zip(alist, tail(iter(alist))))


def lagdiff(alist):
    ialist = iter(alist)
    return list(zipWith(lambda x, y: y - x, ialist, tail(ialist)))


def growth(alist):
    ialist = iter(alist)
    return list(zipWith(lambda x, y: (y - x) / x, ialist, tail(ialist)))
