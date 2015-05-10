#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import Lazy


class Chain(object):

    def __init__(self, cont):
        self.cont = cont

    def bind(self, func):

        if self.cont is None:
            return Chain(None)

        return Chain(func(self.cont))


    def map(self, func):
        if self.cont is None:
            return Chain(None)
        return Chain(map(func, self.cont))

    def select(self, func):
        if self.cont is None:
            return Chain(None)
        return Chain(filter(func, self.cont))

    def reject(self, func):
        if self.cont is None:
            return Chain(None)
        return Chain(filter(lambda x: not func(x), self.cont))


    def att(self, attribute):
        return Chain(map(lambda obj: getattr(obj, attribute), self.cont))

    def key(self, key):
        return Chain(map(lambda obj: obj.get(key), self.cont))

    def method(self, method, *args):
        return Chain(map(lambda obj: getattr(obj, method)(*args), self.cont))


    def count(self):
        return len(self.cont)

    def sum(self):

        if self.cont is None:
            return None
        return sum(self.cont)


    def value(self):
        return self.cont

    def toList(self):
        return Chain(list(self.cont))


    def __str__(self):
        return "Chain : {}".format(self.cont)

    def __repr__(self):
        return "Chain : {}".format(self.cont)

Chain.b = Chain.bind
Chain.m = Chain.map
Chain.s = Chain.select