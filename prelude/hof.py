#!/usr/bin/env python
# -*- coding: utf-8 -*-


@curry
def is_type(type, obj):
    """
    Test if the type of the object.
    
    ReturnS True if the type of the object is type, 
    false otherwise.
    
    :param type: Type of object
    :param obj:  Any python to object.
    :type type:  str 
    :type obj:   undefined
    
    Possible 'type' values
    
    'list', 'tuple', 'num', 'str', 'function', 'none'     
    
    Examples:
    
    >>> from utils3 import hof as h
    >>> h.is_type("none", None)
    True
    >>> h.is_type("num", 34.434)
    True    
    >>> h.is_type("num", 34)
    True
    """
    return is_type.predicate[type](obj)
        
is_type.predicate = {
    'list'  : is_list,
    'tuple' : is_tuple,
    'num'   : is_num,
    'str'   : is_string,
    'function' : is_function,
    'none'  : is_none,
}












@curry
def filterl_index(predicate, List):
    """
    Return the index of elements that satisfy the predicate function.

    :param predicate:
    :param List:
    :return:
    """
    result = []

    for idx, e in enumerate(List):

        if predicate(e):
            result.append(idx)

    return result




@curry
def mapif(predicate, function, list):
    """

    """
    result = list.copy()

    for i, x in enumerate(list):

        if predicate(x):
            result[i] = function(x)

    return result


#-----------------------------------#
#     FUNCTION COMPOSITION          #
#-----------------------------------#

def joinf(funclist):
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

    def joined_functions(x):
        return [f(x) for f in funclist]

    return joined_functions
    #return [ list(map(fi, array)) for fi in funclist ]


def currying(function, variable, **constants):
    """
    :param function:  Function
    :param variable:  Variable symbol (str), free parameter
    :param constants: Constant parameters dictionary
    :return:          Curried function

    Example:

    Create a new function fun_x(x) from f(x)
    such that fun_x(x) = fun(x, y=2, z=3)

    >>> from m2py import functional as f
    >>>
    >>> def fun(x, y, z): return x**2 - y*z
    ...
    >>> fun_x = f.currying(fun, "x", y=2, z=3)
    >>> f.mapl(fun_x, [2, 3, 4, 5, 7])
    [-2, 3, 10, 19, 43]
    """

    params = constants

    def curried_function(x):
        params[variable] = x
        return function(**params)

    return curried_function





def pipe(*funclist):
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

    def f(x):
        _x = x

        for f in funclist:
            _x = f(_x)

        return _x

    return f





def ifelse(condition, trueValue, falseValue):
    """
    :param condition:   Flag
    :param trueValue:   Return value if flag is True
    :param falseValue:  Return value if flag is False
    :type  condition:   Flag (bol value) True/False
    :return:            trueValue if condition is True, FalseValue otherwise

    Example:

    >>> from m2py import functional as f
    >>>
    >>> x = 3
    >>>
    >>> f.ifelse(x> 2, x**2, x-10)
    9
    >>> x= 1
    >>> f.ifelse(x> 2, x**2, x-10)
    -9
    >>> x= 2
    >>> f.ifelse(x> 2, x**2, x-10)
    -8
    >>>

    """
    out = [falseValue, trueValue][condition]

    return out


def ifelsef(condition, trueFunction, falseFunction=identity):
    """
    :param condition:       Condition function
    :param trueFunction:    Function to be executed if condition is True
    :param falseFunction:   Function to be executed if condition is False
    :type condition:        function
    :type trueFunction:     function
    :type falseFunction:    function
    :return:                Conditional function

    Create a new function such that

    function(x):
     if condition(x):
        return trueFun(x)
     else
        return falseFun(x)

    Example:

    Crete
                /  x^2  , if x < 3
        f(x)  =
                \  x/3  , if x >= 3

    >>> from m2py import functional as f
    >>>
    >>> fx = f.ifelsef(lambda x: x<3, lambda x: x**2, lambda x: x/3.0)
    >>>
    >>> f.mapl(fx, [-3, -2, -1, 0, 1, 2, 3, 6, 9, 18, 27])
    [9, 4, 1, 0, 1, 4, 1.0, 2.0, 3.0, 6.0, 9.0]
    >>>
    >>> fx = f.ifelsef(lambda x: x<3, f.constant(0))
    >>> f.mapl(fx, [-3, -2, -1, 0, 1, 2, 3, 6, 9, 18, 27])
    [0, 0, 0, 0, 0, 0, 3, 6, 9, 18, 27]
    >>>
    """

    def func(x):
        return [falseFunction, trueFunction][condition(x)](x)

    return func






def times(function, n):
    """
    Call a function n times

    :param function: Function without argument
    :pram  n:        Number of times to call
    :return:         [y0, y1, y2 ... ]

    for i in range(n):
        yi = function()

    Example:
    >>> fromn functional import ncall
    >>> from random import randint
    >>>
    >>> rnd= lambda : randint(0, 100)
    >>> f.times(rnd, 10)
    [84, 92, 31, 45, 32, 99, 38, 39, 89, 25]
    """
    for i in range(n):
        function()


def groupby(function, sequence):
    """

    Example:

    >>> from m2py import functional as f
    >>> f.groupby(len, ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank'])
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    >>>
    >>>
    >>> f.groupby(f.is_even, [1, 2, 3, 4, 5, 6, 7])
    {False: [1, 3, 5, 7], True: [2, 4, 6]}
    >>>
    """

    output = {}

    for x in sequence:
        y = function(x)

        if not output.get(y):
            output[y] = [x]
        else:
            output[y].append(x)

    return output








def find(predicate, array):
    """
    find_.find(list, predicate, [context]) Alias: detect
    Looks through each value in the list, returning the first one that
    passes a truth test (predicate), or undefined if no value passes the
    test. The function returns as soon as it finds an acceptable element,
    and doesn't traverse the entire list.

    var even = _.find([1, 2, 3, 4, 5, 6], function(num){ return num % 2 == 0; });
    => 2
    """
    for x in array:
        if predicate(x):
            return x
    return None


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








def sublist(list, start, len):
    """
    Returns the sub-list of List1 starting at Start and with (max) Len elements. It is not an error for Start+Len to exceed the length of the list.

    >>> x
    [-1, 2, 10, 23, 23.23]
    >>> f.sublist(x, 1, 3)
    [2, 10, 23]
    >>> f.sublist(x, 2, 3)
    [10, 23, 23.23]
    >>>
    >>>
    """
    return list[start:(start + len)]



def duplicate(N, Elem):
    """

    N = integer() >= 0
    Elem = T
    List = [T]
    T = term()

    Returns a list which contains N copies of the term Elem. For example:

    > lists:duplicate(5, xx).
    [xx,xx,xx,xx,xx]

    Note: Function taken from Erlang -
    http://erldocs.com/17.3/stdlib/lists.html#duplicate
    """
    if hasattr(Elem, "copy"):
        return [Elem.copy() for i in range(N)]
    else:
        return [Elem for i in range(N)]


def copy(object):
    """
    Returns a clone of object

    Note: From cloujure Language
    http://docs.oracle.com/javase/6/docs/api/java/util/Vector.html#indexOf%28java.lang.Object%29
    """

    if hasattr(object, "copy"):
        return object.copy()
    else:
        return object


def append(*ListOfLists):
    """
    ListOfLists = [List]
    List = List1 = [T]
    T = term()

    Returns a list in which all the sub-lists of ListOfLists have been appended. For example:

    > lists:append([[1, 2, 3], [a, b], [4, 5, 6]]).
    [1,2,3,a,b,4,5,6]

    >>> from m2py import functional as f
    >>> f.append([1, 2, 3], ['a', 'b'], [4, 5, 6])
    [1, 2, 3, 'a', 'b', 4, 5, 6]

    Note: Function taken from Erlang -
    http://erldocs.com/17.3/stdlib/lists.html#append
    """
    newlist = []
    mapl(lambda x: newlist.extend(x), ListOfLists)
    return newlist



def memoize(func):
    cache = {}

    def memoized(x):

        out = cache.get(x)
        if not out:
            out = func(x)
            cache[x] = out
            return out
        else:
            return out

    return memoized


def maplx_factory(function):
    """
    Example:

    >>> from m2py import functional as f
    >>>
    >>> def fun(a, b, c): return a**2 - 10*b + c
    ...
    >>>
    >>> fun_v = f.maplx_factory(fun)
    >>>
    >>> fun_v( [(0, 2, 4), (-3, 4, 8), (4, 2, 5), (22, -10, 23)] )
    [-16, -23, 1, 607]
    >>>
    >>> fun_v((0, 2, 4))
    -16
    >>> fun_v((-3, 4, 8))
    -23
    >>> fun_v((4, 2, 5))
    1
    """

    def f(x):
        if is_tuple(x):
            return function(*x)

        elif is_list(x):
            return maplx(function, x)
        raise Exception("Argument must be tuple or List")

    return f

#-----------------------------------#
#          VECTORIZATION            #
#-----------------------------------#

def vectorize_juxt(funclist):
    """

    Example:

    >>> from m2py import functional as f
    >>>
    >>> f1 = lambda x: x**2 - 10.0
    >>> f2 = lambda x: 10*x + 8
    >>> f3 = lambda x: 100.0/x - 4
    >>>
    >>> x = [1, 2, 3, 4, 5]
    >>>
    >>>
    >>> fv= f.vectorize_juxt([f1, f2, f3])
    >>> fv(x)
    [[-9.0, -6.0, -1.0, 6.0, 15.0],
     [18, 28, 38, 48, 58],
     [96.0, 46.0, 29.333333333333336, 21.0, 16.0]]
    """
    f = lambda x: juxt(funclist, x)
    return f


def vectorize(function):
    """
    :param function: Function to be vectorized
    :return:         Vectorized function

    Create new function y= vf(x), such that:

    y = f(x)

        /  map(f, x)  if x is list
    vf =
        \  f(x)       if x is not list

    Example:

    >>> from m2py import functional as f
    >>> import math
    >>>
    >>> sqrt = f.vectorize(math.sqrt)
    >>> sqrt(2)
    1.4142135623730951
    >>>
    >>> sqrt([1, 2, 3, 4, 5])
    [1.0, 1.4142135623730951, 1.7320508075688772, 2.0, 2.23606797749979]
    """

    def vectorized_function(x):
        if isinstance(x, list):
            return list(map(function, x))
        return function(x)

    return vectorized_function


def vectorize_var(function, variable, **constants):
    """
    Create a vectorized curried function

    :param function:  Function
    :param variable:  Variable symbol (str), free parameter
    :param constants: Constant parameters dictionary
    :return:          Curried function


    Example:
    Currying - Create a new function fun_x(x) from f(x)
    such that fun_x(x) = fun(x, y=2, z=3)

    >>> def fun(x, y, z): return x**2 - y*z
    ...
    >>> fun_vx = f.vectorize_var(fun, "x", y=2, z=3)
    >>>
    >>> fun_vx([2, 3, 4, 5, 7])
    [-2, 3, 10, 19, 43]
    >>>
    """
    fv = currying(function, variable, **constants)
    fv = vectorize(fv)
    return fv


def vectorize_dec():
    """
    Vectorize decorator

    Example:

    import functional as f

    @f.vectorize_dec()
    def signum(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    >>> signum([1, 2, -23, 0, -4.23, 23])
    [1, 1, -1, 0, -1, 1]
    >>>
    >>> signum(10)
    1
    >>> signum(-10)
    -1
    >>> signum(0)
    0
    >>>
    """
    import functools

    def wrap(f):
        @functools.wraps(f)
        def wrapper(x):
            if is_list(x):
                return list(map(f, x))
            else:
                return f(x)

        return wrapper

    return wrap






def equalize(*args):
    """
    Similar to copy the non constants elemements
    of an spreadsheet to the other rows.

    x   y   z   w           x   y  z  w
    1   -2  8   6           1  -2  8  6
    2   -1                  2  -1  8  6
    3   0                   3   0  8  6
    4   1          ===>     4   1  8  6
    5   2                   5   2  8  6
    6   3                   6   3  8  6

    Example:

    >>> from m2py.functional.vectorize import equalize
    >>> equalize([1, 2, 3, 4, 5, 6], [-2, -1, 0, 1, 2, 3], 8, 6)
    [[1, 2, 3, 4, 5, 6], [-2, -1, 0, 1, 2, 3], [8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6]]

    Note: I don't have a good name to this function yiet.
    """
    args = list(args)
    v = find(is_list, args)
    if v is not None:
        N = len(v)
        columns = mapif(is_num, lambda x: duplicate(N, x), args)
    else:
        columns = args
    return columns


def vectorize_args(function):
    """
    Creates a new function that accepts
    array or number as input.
    This function is Useful for spreadsheet
    like numerical computations.


    Example:

    f(x , y , z, w) = x*y – 10*z + w

    x   y   z   w   f(x, y, z=8, w=6)
    1   -2  8   6   -76
    2   -1  8   6   -76
    3   0   8   6   -74
    4   1   8   6   -70
    5   2   8   6   -64
    6   3   8   6   -56


    >>> from m2py.functional.vectorize import vectorize_args
    >>> x = [  1,  2, 3, 4, 5, 6 ]
    >>> y = [ -2, -1, 0, 1, 2, 3 ]
    >>> z = 8
    >>> w = 6
    >>>
    >>> def f(x, y, z, w): return x*y - 10*z + w
    ...
    >>>
    >>> mf = vectorize_args(f)
    >>>
    >>> mf(x, y, z, w)
    [-76, -76, -74, -70, -64, -56]
    >>>
    """

    def func(*args):
        columns = equalize(*args)
        return maplx(function, list(zip(*columns)))

    return func

#-------------------------#
#        MATRIX           #
#-------------------------#

@curry
def get_column(column, rows):
    """
    Get the column of a matrix ( List ) composed
    of list or tuple that represents each row of
    a matrix.

    :param matrix:  List cotaining [ row0, row1, ... rown]  where row_i = [ ai0, ai1, ai2, ... ain]
    :param column:  Column number ( Example: k to get column k)
    :return:        Column k or  [ a0k, a1k, a2k, ... aMk ]

    Example:

    x   y   z
    5   43  83
    52  99  70
    78  27  86
    26  84  49


    Represented as:

    [
    (x0, y0, z0),
    (x1, y2, z1),
    (x2, y2, z2),
    ...
    (xn, yn, zn)
    ]

    [(5,  43, 83),
     (52, 99, 70),
     (78, 27, 86),
     (26, 84, 49),]

    Each List
    >>> M = [(5.0, 52.0, 78.0, 26.0), (43.0, 99.0, 27.0, 84.0), (83.0, 70.0, 86.0, 49.0)]
    >>> get_column(M, 0)
    [5.0, 43.0, 83.0]
    >>> get_column(M, 1)
    [52.0, 99.0, 70.0]
    >>> get_column(M, 2)
    [78.0, 27.0, 86.0]
    >>> get_column(M, 3)
    [26.0, 84.0, 49.0]

    """

    return list(map(lambda row: row[column], rows))

@curry
def map_rows(function, rows):
    """
    Map a function to each row in matrix of rows.
    
    Example:
    
    >>> map_rows(add(100), [(10, 20), (30, 50), (60, 80)])
    [[110, 120], [130, 150], [160, 180]]   
    """
    return  list(map(lambda row: list(map( lambda element: function(element), row)), rows  ))


