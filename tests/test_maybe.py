#!/usr/bin/env python3
"""
>>> Nothing >> safediv(9)
    Nothing

>>> Just(10) >> safediv(9)
    Just(1.1111111111111112)

>>> Just(10) >> safediv(9) >> safediv(10)
    Just(0.11111111111111112)

>>> Just(10) >> safediv(9) >> safediv(10) >> safediv(8)
    Just(0.01388888888888889)

>>> Just(10) >> safediv(9) >> safediv(0) >> safediv(8)
    Nothing

>>> 


"""

from prelude import curry
from prelude.monad import Nothing, Maybe, Just
from prelude.Dict import lookup, lookup_nested

@curry
def safediv(x, y):
    
    if x == 0:
        return Nothing
    else:
        return Just(y/x)
    

data = {"foo": {"bar": {"baz": "bing"}}}
test_1 = lookup("foo", data).bind(lookup("bong")).bind(lookup("baz")) == Nothing
test_2 = lookup_nested(["foo", "bong", "baz"], data) == Nothing
test_3 = lookup_nested(["foo", "bar", "baz"], data)  == Just("bing")

test_4 = lookup("foo", data) >> lookup("bong") >> lookup("baz") ==  Nothing

test_5 = lookup("foo", data) >> lookup("bar") >> lookup("baz") == Just('bing')

test_6 = Nothing >> lookup("foo") >> lookup("1bar") >> lookup("baz") == Nothing

test_7 = Just(data) >> lookup("foo") >> lookup("bar") >> lookup("baz") == Just('bing')

try:
    assert all([test_1, test_2, test_3, test_4, test_5, test_6, test_7])
    print("Passed: All tests OK!")
    
except:
    print ("Tests Failed")
