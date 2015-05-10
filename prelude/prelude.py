"""
#
#
#   Functional Programming Features:
#
#   map
#   filter, 
#   functools.reduce
#
#
#
#
"""
from functools import reduce
import itertools
import time


class CurriedFunction(object):
    """
    Curried Function Class
    
    To see function documentation, enter
        
        self.help()
    """

    def __init__(self, funct, argvalues=[]):

        self._funct = funct
        self._nargs = self._funct.__code__.co_argcount
        #self._nargs = self._funct.__code__.nlocals
        self._argvs = argvalues

        # setattr(self, "__doc__", _funct.__doc__)

        self.__doc__ = funct.__doc__

    def __call__(self, *args):

        argvalues = self._argvs.copy()
        argvalues.extend(args)

        # print(_argvs)
        #print(self._nargs)

        if len(argvalues) == self._nargs:

            return self._funct(*argvalues)

        elif len(argvalues) > self._nargs:
            raise TypeError("Wrong number of arguments")

        else:
            newfunctor = CurriedFunction(self._funct, argvalues)
            return newfunctor



def curry(function):
    return CurriedFunction(function)


to_list = list  # Convert generator to list
to_tuple = tuple  # Convert gernerator to tuple
to_stream = iter  # Convert iterable object to generator

reverse = reversed
reversel = lambda alist: list(reversed(alist))

#############################
# TYPE CHECKING           #
#############################



def is_list(var):
    """
    Test if variable var is list

    :return: True if var is list, False if var is not list
    :rtype:  bol
    """
    return isinstance(var, list)


def is_tuple(var):
    return isinstance(var, tuple)


def is_num(var):
    """
    Test if variable var is number (int or float)

    :return: True if var is number, False otherwise.
    :rtype:  bol
    """
    return isinstance(var, int) or isinstance(var, float)

def is_int(var):
    return isinstance(var, int)

def is_float(var):
    isinstance(var, float)

def is_dict(var):
    return isinstance(var, dict)


def is_str(var):
    return isinstance(var, str)


def is_function(var):
    """
    Test if variable is function (has a __call__ attribute)

    :return: True if var is function, False otherwise.
    :rtype:  bol
    """
    return hasattr(var, '__call__')


def is_none(var):
    return var is None


def is_empty(lst):
    return not lst


def is_gen(x):
    return hasattr(x, "__next__")


#############################
#   OPERATORS               #
#############################


@curry
def add(x, y):
    return x + y


@curry
def sub(x, y):
    return y - x


@curry
def div(x, y):
    return y / x

@curry
def divi(x, y):
    return y // x

@curry
def mod(x, y):
    return y % x

@curry
def mul(x, y):
    return x * y


@curry
def pow(x, y):
    return x ** y


@curry
def contains(lst, value):
    return value in lst


@curry
def lt(x, y):
    """
    leq :: a -> a -> bool
    Less than
    
    :return: (x < y)            
    """
    return x < y
    
@curry 
def le(x, y):
    """
    leq :: a -> a -> bool
    Less or equal
    
    :return: (x <= y)
    """
    return x > y

@curry
def gt(x, y):
    """
    gt :: a -> a -> bool
    Greater than
    
    :return: x > y
    """
    return x > y
    

@curry
def ge(x, y):
    """
    Greater or equal
    gt :: a -> a -> bool
      
    :return: x >= y
    """
    return x >= y

@curry
def eq(x, y):
    """
    Equal
    eql :: a -> a -> bool
    
    :return: x == y
    """
    return x == y
    
@curry 
def neq(x, y):
    """
    Not equal
    neq :: a -> a -> bool
    
    :return: x != y
    """


def identity(x):
    """
    Identity fucntion:
    
    identity :: a -> a
    identity x = x
    """
    return x


def constant(a):
    """
    Constant function
    
    constant :: a -> b -> a
    constant(a, b) = a
    """
    return lambda b: a


def uncurry(function):
    return lambda atuple: function(*atuple)


flip = lambda f: lambda x, y: f(y, x)

zipl = lambda *atuple: list(zip(*atuple))

@curry
def allmap(predicate, iterable):
    return all(map(predicate, iterable))


@curry
def anymap(predicate, iterable):
    return any(map(predicate, iterable))


@curry
def mapf(function, stream):
    return map(function, stream)


@curry
def filterf(function, stream):
    return filter(function, stream)


@curry
def filterl(function, stream):
    return list(filter(function, stream))


@curry
def mapl(function, stream):
    return list(map(function, stream))


@curry
def foldl(f, x0, alist):
    """
    In [51]: foldr(lambda a, b: a + b*16.0, 0, [13, 0, 1, 3])
    Out[51]: 12557.0
    
    λ> foldr (\b c -> b + 16*c) 0 [13, 0, 1, 3] 
    12557
    λ> 
    """
    acc = x0

    for x in alist:
        acc = f(acc, x)
    return acc


@curry
def foldr(f, x0, alist):
    """
    In [51]: foldr(lambda a, b: a + b*16.0, 0, [13, 0, 1, 3])
    Out[51]: 12557.0
    
    λ> foldr (\b c -> b + 16*c) 0 [13, 0, 1, 3] 
    12557
    λ> 
    """
    return reduce(lambda x, y: f(y, x), reversed(alist), x0)


@curry
def foldl1(f, alist):
    return reduce(f, alist)


@curry
def foldr1(f, alist):
    return reduce(lambda x, y: f(y, x), reversed(alist))


@curry
def starmap(function, arglist):
    """
    map tuple
    
    Map list of function arguments to the function

    :param function: Function of tuples
    :param arglist:  List of arguments of f
    :return:         list of results

    Let be
        function: f( a, b, c, d, ...)
        arglist :  [ (a0, b0, c0, ...), (ak, bk, ck, ...), ... ]

        Xk = [ ak, bk, ck, ... ]
        return [ f(X0), f(X1), ... f(Xn)]

    Example:

    >>> from m2py import functional as f
    >>>
    >>> x= [ (0, 2, 4), (-3, 4, 8), (4, 2, 5), (22, -10, 23)]
    >>>
    >>> def fun(a, b, c): return a**2 - 10*b + c
    ...
    >>>
    >>> f.maplx(fun, x)
    [-16, -23, 1, 607]
    """
    return (function(*params) for params in arglist)


def entryPoint(function):
    """
    Decorator to set the function
    as the main function.
    
    Example:
    
    @mainf
    def myfunction():
        print("Hello World")
    
    """

    if __name__ == "__main__":
        function()


def profile(function):
    """
    Measuere the execution time.
    of a function.
    """

    def _(*args, **kwargs):
        tstart = time.time()
        result = function(*args, **kwargs)
        tend = time.time()

        print("Time taken: ", (tend - tstart))
        return result

    return _

def cpipe(*funclist):
    """
    Compose a list of functions

    f = cpipe (f1, f2, f3, f4)

    f(x) = f4( f3( f2( f1 x ))))

    :param funclist:
    :return:
    """

    def _(args):
        value = args

        for f in funclist:

            #print("value = ", value)

            value = f(value)

        return value

    return _


def compose(*funclist):
    """
    Returns the composition of a list of functions, where each function
    consumes the return value of the function that follows. In math terms,
    composing the functions f()

    :param funclist: List of functiosn [f0, f1, f2, f3, ... fn-1]
    :return:         New function f(x) = f0(f1(...fn-3(fn-2(fn-1(x))))
    :type funclist:  list(function)
    :rtype funclist: function

    Create f(x) such that

    f(x) = (f0.f1.f2...fn-1)(x)


    Example:

    Compute inc(double(10)) = 21

    >>>
    >>> imoort functional as f
    >>>
    >>> inc = lambda x: x+1
    >>> double = lambda x: x*2
    >>>
    >>> f.compose(inc, double)(10)
    21
    """
    flist = list(funclist)
    flist.reverse()

    def _(x):
        _x = x

        for f in flist:
            _x = f(_x)

        return _x

    return _


def juxt(funclist):
    """
    Map a list of functions to an array

    :param funclist: List of functions  [ f0, f1, f2 .. fk]
    :param array:    List of values     x= [ x0, x1, x2, ... xn]
    :return:         [map(f0, x), map(f1, x), ... map(fn, x)]


    juxt takes two or more functions and returns a function that returns
    a vector containing the results of applying each function on its
    arguments. In other words, ((juxt a b c) x) => [(a x) (b x) (c x)].
    This is useful whenever you want to represent the results of using 2
    different functions on the same argument(s), all at once rather than separately:

    Example:

    >>> from m2py import functional as f
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>> fun1 = lambda x: x**2 - 10.0
    >>> fun2 = lambda x: 10*x + 8
    >>> fun3 = lambda x: 100.0/x - 4
    >>> fun= [fun1, fun2, fun3 ]
    >>> f.joinfuncs(fun, x)
    [[-9.0, -6.0, -1.0, 6.0, 15.0],
     [18, 28, 38, 48, 58],
     [96.0, 46.0, 29.333333333333336, 21.0, 16.0]]


    Note: Function taken from cloujure and R
    https://clojuredocs.org/clojure.core/juxt
    """

    def _(x):
        return [f(x) for f in funclist]

    return _

def mcall(method):
    """
    Call a method from an object:
    """
    def _(*args, **kwargs):
        return lambda obj: getattr(obj, method)(*args, **kwargs)
    return _


def sliding_window(array, k):
    """
    A sequence of overlapping subsequences

    Example:

    >>>
    >>> x = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    >>>
    >>> print(f.sliding_window(x, 1))
    [('x0',), ('x1',), ('x2',), ('x3',), ('x4',), ('x5',), ('x6',), ('x7',), ('x8',), ('x9',)]
    >>>
    >>> print(f.sliding_window(x, 2))
    [('x0', 'x1'), ('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x5'), ('x5', 'x6'), ('x6', 'x7'), ('x7', 'x8'), ('x8', 'x9')]
    >>>
    >>> print(f.sliding_window(x, 3))
    [('x0', 'x1', 'x2'), ('x1', 'x2', 'x3'), ('x2', 'x3', 'x4'), ('x3', 'x4', 'x5'), ('x4', 'x5', 'x6'), ('x5', 'x6', 'x7'), ('x6', 'x7', 'x8'), ('x7', 'x8', 'x9')]
    >>>

    Note: http://toolz.readthedocs.org/en/latest/api.html#toolz.itertoolz.sliding_window
    """
    return zip(*[array[i:] for i in range(k)])





@curry
def nth(alist, n):
    """
    Get the nth element of a list or tuple.
    
    nth :: [a] -> int -> a
    nth(alist, n) = alist[n]
    """
    return alist[n]

@curry 
def nths(n, alist):
    """
    Get the nth element of a list or tuple.
        
    nth :: int -> [a] -> a
    nth (n, alist) = alist[n]    
    """    
    return alist[n]


    
@curry
def column_rows(array_of_rows, n):
    return list(map(lambda row: row[n], array_of_rows))

@curry
def column_nth(n, array_of_rows):
    return list(map(lambda row: row[n], array_of_rows))

@curry
def getat(attribute, object):
    return getattr(object, attribute)
    

@curry
def slice(alist, i1, i2):
    """
    slice the same as alist[i1:i2]
    
    slice :: [a] -> int -> int -> a
    
    """
    return alist[i1:i2]


def delaycall(function, args=(), kwargs=None):
    """
    """
    if not kwargs:
        kwargs = {}
    return lambda: function(*args, **kwargs)


def call(function, args=(), kwargs=None):
    if not kwargs:
        kwargs = {}
    return function(*args, **kwargs)


def unique(lst):
    """
    Remove repeated elements from a list
    
    unique :: [a] -> [a]
    """
    return sort(set(lst))



def find_index(predicate, List):
    """
    (a → Boolean) → [a] → [Number]

    Return the index of first element that satisfy the
    predicate
    """
    for i, x in enumerate(List):
        if predicate(x):
            return i


def find_indices(predicate, List):
    """
    Returns an array of all the indices of the
    elements which pass the predicate. Returns an
    empty list if the predicate never passes.
    find-indices even, [1 2 3 4] #=> [1, 3]

    >>> find_indices(lambda x: x > 2, [1, 2, 30, 404, 0, -1, 90])
    [2, 3, 6]
    """
    result = []

    for i, x in enumerate(List):
        if predicate(x):
            result.append(i)

    return result
