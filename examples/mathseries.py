#!/usr/bin/env python
# -*- coding: utf-8 -*-

#---------------------------------------------------#

from prelude import last, take, iterate

def until_converge(stream, eps, itmax):

    a = next(stream)
    b = next(stream)

    n = 1

    while True:

        if n > itmax:
            break

        if abs((b-a)/a) < eps:
            break

        yield a
        yield b

        a = b
        b = next(stream)
        n = n + 1

    print("n = ", n)

def summation(stream):

    s = 0
    while True:
        s += next(stream)
        yield s

def aitken(stream):
    Sn  = next(stream)
    Sn1 = next(stream)
    Sn2 = next(stream)


    while True:

        s = Sn2 - (Sn2 - Sn1)**2 / (Sn2 - 2*Sn1 + Sn)
        Sn, Sn1, Sn2 = Sn1, Sn2, next(stream)
        yield s

def infinite_serie(term, inext, start):

    i = start

    while True:
        yield term(i)
        i = inext(i)


def fixed_point(iterator, eps, itmax, guess):

    stream = iterate(iterator, guess)
    return last(until_converge(stream, eps, itmax))


def nsolver(f, df, guess, eps, itmax):

    iterator = lambda x: x - f(x)/df(x)
    return fixed_point(iterator, eps, itmax, guess)


def pi_serie():

    s = 0
    i = 1
    k = 0

    while True:

        s = s + i*4/(2*k+1)
        i = -1 * i
        k = k + 1
        yield s


def fib(n):

    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


def root(a):
    f = lambda x: 0.5 * (a / x + x)
    return last(take(10, iterate(f, 1.0)))
