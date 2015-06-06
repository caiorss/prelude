"""

prelude

Author: Caior Rodrigues Soares Silva - 2015

Description:
    A functional Programming Library based on Haskell prelude.hs
    This module provides many useful curried functions and higher order
    functions to manipulate iterators and lists.

    The functions are described in Haskell notation.

Operators functions:
-------------------------
        
This functions are useful to be passed as an argument to map or any
other higher order function.

f :: a -> a -> a

add (x, y) = x + y
sub (x, y) = x - y
div (x, y) = x / y
mul (x, y) = x * y
pow (x, y) = x ** y

f :: a -> a -> bool

neq (x, y)  = x != y     Not Equal
eq  (x, y)  = x == y     Equal
lt  (x, y)  = x <  y     Less Than
le  (x, y)  = x <= y     Less or equal than
gt  (x, y)  = x >  y     Greater
ge  (x, y)  = x >= y     Greater or Equal

      

Type Checking functions:
------------------------

is_num      :: val -> bool     Test if value is number (float or int)
is_int      :: val -> bool     Test if value is an int
is_float    :: val -> bool     Test if value is a  float

is_dict     :: val -> bool     Test if value is dictionary
is_function :: val -> bool     Test if value is a function
is_gen      :: val -> bool     Test if value is a a generator
is_none     :: val -> bool     Test if value is None

is_tuple    :: val -> bool
is_list     :: val -> bool
is_empty    :: val -> bool      Test if is an empty list or tuple


Functions to Operate over Lists
-------------------------

nth :: [a] -> int -> a          Get the nth element of a list or tuple.
nth(alist, n) = alist[n]        Useful to get many elements from same list
                            

nth :: int -> [a] -> a          Get the nth element of a list or tuple.
nth (n, alist) = alist[n]       Useful to get the elements of same order of many lists.
                                
contains :: [x] -> x -> bool    Returns True if the element is in the list
contains(xs, x) = x in xs

slice :: [a] -> int -> int -> a
slice(alist, i1, i2) = alist[i1:i2]


unique :: [a] -> [a]            Remove repeated elements from a list
unique(alist)

Higher Order Functions
-------------------------      

Identity function

    identity :: a -> a
    identity x = x

Constante Function
    
    constant :: a -> b -> a
    constant(a, b) = a

Returns a new function with inverted arguments

    flip :: (x -> y -> z) -> (y -> x -> z)  
    flip (f :: x -> y -> z) = g :: y -> x -> z
    
    Example:
    
    >>> import prelude as p
    >>> p.sub(10, 100)
        90
    >>> p.flip(p.sub)(10, 100)
        -90


Special Functions 
-------------------------  
    
    entryPoint
    ---------------------
        
    Intestead of write:
    
        def main():
            print ("This is the main function")
            
        if __name__ == "__main__":
            main()
            
    Just add the entryPoint decorator            
        
        @entryPoint      --> Set the main function
        def main(x):
            print ("This is the main function")
    
    profile
    ---------------------
        Measure the function execution time.
        
        profile(someFunction)(Function arguments)



************************************************

Examples:

>>> import prelude as p

>>> p.mapl(p.add(20), range(10))
[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

>>> # Higher order funtions ending in  retuns a list
>>> p.mapl(p.sub(20), range(10, 30))
    [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> # Higher order funtions ending in f retuns a generator/stream object
>>>  p.mapf(p.sub(20), range(10, 30))

>>> # Filter elements greater than zero le(0) = 
>>> # (\\ x y -> x < y) 0 y ==:> (\\ y -> 0 < y) y
>>>
>>> p.filterl(p.le(0), p.mapf(p.sub(20), range(10, 30)))
    [1, 2, 3, 4, 5, 6, 7, 8, 9]


>>> p.filterl(p.gt(0), p.mapf(p.sub(20), range(10, 30)))
    [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]


"""

__author__ = "Caio Rodrigues Soares Silva"
__email__ = "caiorss.rodrigues@gmail.com"


from .Operator import X
# from . import Lazy
from .Chain import Chain
from .ControlFlow import ifelse, ifelseDo
from . import Str
from . import monad


from .prelude import mapl, filterl, compose, cpipe, juxt, identity, constant
from .prelude import zipl, mapf, mapl, filterf, filterl
from .prelude import foldr, foldl, foldr1, foldl1

# Special Functios and Objects
from .prelude import profile, entryPoint
# Stream Functions
from .prelude import curry, uncurry, to_list, reverse, reversel, mcall, mcallm
# Type Checking Functions
from .prelude import (is_num, is_int, is_float, is_dict, is_str,
                      is_function, is_empty, is_gen, is_none, is_tuple, is_list)

# Operator Functions
from .prelude import (add, mul, sub, div, mod, divi, pow, contains,
                      nth, nths, column_nth, column_rows, slice, attrib, flip)

# Comparator predicate functions
from .prelude import lt, le, gt, ge, eq, neq

from .prelude import unique


from .prelude import column_nth, column_rows, tail, init


from .prelude import readFile, writeFile
