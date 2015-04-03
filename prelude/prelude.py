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
>>> piserie  = summation(infinite_serie(lambda k: 8/(k*(k+2)), lambda i: i+4, 1))

>>> last(converge(piserie, 1e-6, 1e9))
n =  399
    3.14034


>>> piserie  = summation(infinite_serie(lambda k: 8/(k*(k+2)), lambda i: i+4, 1))

>>> last(converge(aitken(piserie), 1e-10, 1e9))
n =  6200
    3.14155

>>> piserie  = summation(infinite_serie(lambda k: 8/(k*(k+2)), lambda i: i+4, 1))

>>> last(converge(piserie, 1e-10, 1e9))n =  39894
    3.14158



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


def help2(obj):
    print(obj.__doc__)


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
def sub(y, x):
    return y - x


@curry
def div(y, x):
    return y / x


@curry
def mul(x, y):
    return x * y


@curry
def pow(x, y):
    return x ** y


@curry
def contains(lst, value):
    return value in lst


# curry :: ((a, b) -> c) -> a -> b -> c

def identity(x):
    return x


def constant(constant):
    return lambda *args, **kwargs: constant


def uncurry(function):
    return lambda atuple: function(*atuple)


flip = lambda f: lambda x, y: f(y, x)


def zipWith(func, stream1, stream2):
    return map(lambda e: func(e[0], e[1]), zip(stream1, stream2))


def take(n, iterable):
    for i in range(n):

        try:
            yield next(iterable)
        except:
            pass


def takeWhile(predicate, stream):
    while True:

        x = next(stream)

        if not predicate(x):
            break

        try:
            yield x
        except StopIteration:
            break


def dropWhile(predicate, stream):
    while True:

        x = next(stream)

        if not predicate(x):
            try:
                yield x
            except StopIteration:
                break


def flat(stream):
    return itertools.chain(*stream)


@curry
def flat_mapl(function, stream):
    return flat(map(function, stream))


def iterate(f, x):
    y = x

    while True:

        try:
            yield y
            y = f(y)
        except StopIteration:
            break


def tail(stream):
    next(stream)
    return stream


def last(stream):
    return next(reversed(tuple(stream)))


def head(stream):
    return next(stream)


def drop(n, stream):
    for i in range(n):
        next(stream)

    return stream


def foreach(function, iterable):
    for element in iterable:
        function(next(iterable))




def pairs(alist):
    return zip(alist, tail(iter(alist)))


def pairsl(alist):
    return list(zip(alist, tail(iter(alist))))


def lagdiff(alist):
    ialist = iter(alist)
    return to_list(zipWith(lambda x, y: y - x, ialist, tail(ialist)))


def growth(alist):
    ialist = iter(alist)
    return to_list(zipWith(lambda x, y: (y - x) / x, ialist, tail(ialist)))


def infinite(start=0):
    idx = start

    while True:
        yield idx
        idx += 1


def infinite_alternate(start=0):
    idx = start
    sig = 1

    while True:
        yield idx * sig
        idx += 1
        sig *= -1


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
    return reduce(lambda x, y: f(y, x), reversed(alist), x0)


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


def mainf(function):
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

def compose_pipe(*funclist):
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

    >>> from m2py import functional as f
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
def nths(stream, n):
    """

    nth(N, List) -> Elem

    > lists:nth(3, [a, b, c, d, e]).
    c

    Idea from: http://erldocs.com/17.3/stdlib/lists.html
    """

    i = 0
    while i < n:
        next(stream)  # Consume stream
        i += 1
    return next(stream)


@curry
def nth(alist, n):
    return alist[n]


@curry
def slice(alist, i1, i2):
    return alist[i1:i2]


def retry(call, tries, errors=Exception):
    for attempt in range(tries):
        try:
            return call()
        except errors:
            if attempt + 1 == tries:
                raise


def ignore(call, errors=Exception):
    try:
        return call()
    except errors:
        return None


def in_sequence(function_list):
    """
    Create a new function that execute the functions
    in the list in sequence.

    :param function_list: List of functions
    :return:              Function

    """

    def seqfun():
        for f in function_list: f()

    return seqfun


def in_parallel(function_list):
    """
    Create a new function that execute the functions
    in the list in parallel (thread)

    :param function_list: List of functions
    :return:              Function

    Example:

    >>> from m2py import functional as funcp
    >>>
    >>> import time
    >>>
    >>> def print_time(thname, delay):
    ...   for i in range(5):
    ...      time.sleep(delay)
    ...      print (thname, " ", time.ctime(time.time()))
    >>>
    >>> def make_print_time(name, delay): return lambda : print_time(name, delay)
    >>>
    >>> t1 = make_print_time("thread1", 1)
    >>> t2 = make_print_time("thread2", 2)
    >>> t3 = make_print_time("thread3", 3)
    >>> t4 = make_print_time("thread4", 4)
    >>>
    >>> thfun = funcp.in_parallel([t1, t2, t3, t4])
    >>> thfun()
    >>> thread1   Fri Dec 26 23:40:29 2014
    thread2   Fri Dec 26 23:40:30 2014
    thread1   Fri Dec 26 23:40:30 2014
    thread3   Fri Dec 26 23:40:31 2014
    thread1   Fri Dec 26 23:40:31 2014
    thread4   Fri Dec 26 23:40:32 2014
    thread2   Fri Dec 26 23:40:32 2014
    thread1   Fri Dec 26 23:40:32 2014
    thread1   Fri Dec 26 23:40:33 2014
    ...
    """

    def pfun():
        for func in function_list:
            thread.start_new_thread(func, ())

    return pfun


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
    Remove repeated elements from an aray
    """
    return sort(set(lst))


#---------------------------------------------------#

def to_value():
    """ Dummy function """
    pass


class Stream(object):
    """
    Stream Monad ( Equivalent to List monaf)
    
    >>>  Stream(10.23) >>  (lambda x: x**2) >> (lambda y: y/10) >> to_value 
    10.46529
    
    >>>  Stream(range(8)) >>  mapf(lambda x: x**2) >> mapf(lambda y: y/10) >> to_list 
    [0.00000, 0.10000, 0.40000, 0.90000, 1.60000, 2.50000, 3.60000, 4.90000]
    
    
    >>>  Stream(range(8)) >>  mapf(add(10)) >> mapf(mul(8.5)) >> to_list    [85.00000,
    93.50000,
    102.00000,
    110.50000,
    119.00000,
    127.50000,
    136.00000,
    144.50000]

    
    """

    def __init__(self, value=None):
        self.value = value

    def bind(self, function):
        return Stream(function(self.value))

    def __rshift__(self, other):


        if other == to_value:
            return self.value

        if other == list:
            return list(self.value)

        elif hasattr(other, "__call__"):
            #p = Pipe(list(map(other, self.value)))
            p = Stream(other(self.value))

        else:
            p = Stream(other)

        return p

    def __lshift__(self, other):

        return Stream(other)


class Operator():
    """
    Operator to Generate Lambda expressions

    Scala-style lambdas definition

    Idea from: https://github.com/kachayev/fn.py#fnpy-enjoy-fp-in-python

    Example:

    In [1]: from functional import X, mapl, filterl

    In [2]: list(filter(X  < 10, [9, 10, 11]))
    Out[2]: [9]

    In [4]: mapl(X ** 2, [1, 2, 3, 4, 5, 6, 7])
    Out[4]: [1, 4, 9, 16, 25, 36, 49]

    In [2]: mapl(X  / 10, [9, 10, 11])
    Out[2]: [0.9, 1.0, 1.1]

    In [3]:  mapl( 10/X, [9, 10, 11])
    Out[3]: [1.1111111111111112, 1.0, 0.9090909090909091]

    """

    """
    Scala-style lambdas definition

    Idea from: https://github.com/kachayev/fn.py#fnpy-enjoy-fp-in-python

    Example:

    In [1]: from functional import X, mapl, filterl

    In [2]: list(filter(X  < 10, [9, 10, 11]))
    Out[2]: [9]

    In [4]: mapl(X ** 2, [1, 2, 3, 4, 5, 6, 7])
    Out[4]: [1, 4, 9, 16, 25, 36, 49]

    In [2]: mapl(X  / 10, [9, 10, 11])
    Out[2]: [0.9, 1.0, 1.1]

    In [3]:  mapl( 10/X, [9, 10, 11])
    Out[3]: [1.1111111111111112, 1.0, 0.9090909090909091]

    """

    def __add__(self, other):

        if isinstance(other, Operator):
            return lambda x, y: x + y
        return lambda x: x + other

    def __radd__(self, other):
        if isinstance(other, Operator):
            return lambda x, y: x + y
        return lambda x: other + x

    def __mul__(self, other):

        if isinstance(other, Operator):
            return lambda x, y: x * y
        return lambda x: x * other

    def __rmul__(self, other):

        if isinstance(other, Operator):
            return lambda x, y: x * y
        return lambda x: x * other


    def __sub__(self, other):
        return lambda x: x - other

    def __rsub__(self, other):
        return lambda x: other - x


    def __div__(self, other):
        return lambda x: x / other

    def __truediv__(self, other):
        return lambda x: x / other

    def __floordiv__(self, other):
        return lambda x: x // other

    def __rdiv__(self, other):
        return lambda x: other / x

    def __rtruediv__(self, other):
        return lambda x: other / x

    def __rfloordiv__(self, other):
        return lambda x: other // x

    def __pow__(self, other):
        return lambda x: x ** other

    def __rpow__(self, other):
        return lambda x: other ** x

    def __neg__(self):
        return lambda x: -x

    def __pos__(self):
        return lambda x: x

    def __abs__(self):
        return lambda x: abs(x)

    def __len__(self):
        return lambda x: len(x)

    def __eq__(self, other):
        return lambda x: x == other

    def __ne__(self, other):
        return lambda x: x != other

    def __lt__(self, other):
        return lambda x: x < other

    def __le__(self, other):
        return lambda x: x <= other

    def __gt__(self, other):
        return lambda x: x > other

    def __ge__(self, other):
        return lambda x: x >= other

    def __or__(self, other):
        return lambda x: x or other

    def __and__(self, other):
        return lambda x: x and other

    def __rand__(self, other):
        return lambda x: other and x

    def __ror__(self, other):
        return lambda x: other or x

    def __contains__(self, item):
        return lambda x: item in x

    def __int__(self):
        return lambda x: int(x)

    def __float__(self):
        return lambda x: float(x)

    def split(self, pattern=' '):
        return lambda x: x.split(pattern)

    def strip(self):
        return lambda x: x.strip()

    def map(self, function):
        return lambda x: list(map(function, x))

    def sum(self):
        return lambda x: sum(x)

    def key(self, keyname):
        """Generate lambda expression for dictionary key """
        return lambda x: x[keyname]

    def item(self, it):
        """Generate lambda function for list item """
        return lambda x: x[it]


X = Operator()


