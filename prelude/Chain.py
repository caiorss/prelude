#!/usr/bin/env python
# -*- coding: utf-8 -*-


from . import Lazy


class Chain(object):

    def __init__(self, content):
        self.content = content

    def bind(self, func):

        if self.content is None:
            return Chain(None)

        return Chain(func(self.content))


    def map(self, func):
        if self.content is None:
            return Chain(None)
        return Chain(map(func, self.content))

    def select(self, func):
        if self.content is None:
            return Chain(None)
        return Chain(filter(func, self.content))

    def reject(self, func):
        if self.content is None:
            return Chain(None)
        return Chain(filter(lambda x: not func(x), self.content))


    def att(self, attribute):
        return Chain(map(lambda obj: getattr(obj, attribute), self.content))

    def key(self, key):
        return Chain(map(lambda obj: obj.get(key), self.content))

    def method(self, method, *args):
        return Chain(map(lambda obj: getattr(obj, method)(*args), self.content))


    def count(self):
        return len(self.content)

    def sum(self):

        if self.content is None:
            return None
        return sum(self.content)


    def value(self):
        return self.content

    def toList(self):
        return Chain(list(self.content))

    def __rshift__(self, function):
        return Chain(function(self.content))

    def __str__(self):
        return "Chain : {}".format(self.content)

    def __repr__(self):
        return "Chain : {}".format(self.content)

Chain.b = Chain.bind
Chain.m = Chain.map
Chain.s = Chain.select
