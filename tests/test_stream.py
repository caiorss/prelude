#!/usr/bin/env python
# -*- coding: utf-8 -*-

from prelude import Stream, to_list, to_value, mapl, mapf, compose_pipe


func_pipe1 = lambda x: Stream() >> x \
                       >> mapf(lambda x: x ** 2) \
                       >> mapf(lambda y: y / 10) \
                       >> to_list

func_pipe2 = compose_pipe(
    iter,
    mapf(lambda x: x ** 2),
    mapf(lambda y: y / 10),
    to_list,
)

assert func_pipe1(range(8)) == [0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9]
assert func_pipe2(range(8)) == [0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9]