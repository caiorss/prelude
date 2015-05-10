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

    #print("n = ", n)





def fixed_point(iterator, eps, itmax, guess):

    stream = iterate(iterator, guess)
    return last(until_converge(stream, eps, itmax))


def nsolver(f, df, guess, eps, itmax):

    iterator = lambda x: x - f(x)/df(x)
    return fixed_point(iterator, eps, itmax, guess)



def fib(n):

    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


def root(a):
    f = lambda x: 0.5 * (a / x + x)
    return last(take(10, iterate(f, 1.0)))
