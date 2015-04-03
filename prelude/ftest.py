#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functional Programming Testing Suite

Provides higher order functions to create test cases.

"""

def counter():
    n = 0

    def _():
        nonlocal n
        n += 1
        return n
    return _



def test_case(function, case, expected):

    def _():


        try:
            assert function(case)  == expected
        except AssertionError:
            print("Test Failed")

            print("expected :", expected)
            print("returned :", function(case))

            return 0


        print("test OK")
        return 1

    return _

def run_tests(*list_of_tests):

    for test in list_of_tests:
        test()