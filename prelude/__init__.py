"""
prelude

"""

from .prelude import (head, tail, last, drop, foreach, flat, 
    mapl, filterl, compose, identity, constant)

from .prelude import mapf, mapl, filterf, filterl, zipWith
from .prelude import foldr, foldl, foldr1, foldl1, compose_pipe

# Special Functios and Objects
from .prelude import profile, mainf
from .prelude import Operator 

# Stream Functions
from .prelude import curry, uncurry, Stream, to_value, to_list, reverse, reversel, methodcaller

# Predicate Functions
from .prelude import is_num, is_dict, is_str, is_function, is_empty, is_gen, is_none, is_tuple, is_list

# Operator Functions
from .prelude import add, mul, sub, div, pow, contains, nth, nths, slice




# String Functions
#
from .string import joinstr, splitstr, has_suffix, has_prefix, strip 

from .monad import Maybe, Just, Nothing

X = Operator()
#import .prelude
#import .monad
#import .string
#import .dic
