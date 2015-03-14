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

import time


class CurriedFunction(object):
    
    def __init__(self, funct, argvalues=[]):
    
        self.funct = funct
        self.nargs = self.funct.__code__.co_argcount
        self.argvalues = argvalues
        #self.__doc__ = f.__doc__
    
    def __call__(self, *args):
              
        argvalues = self.argvalues.copy()
        argvalues.extend(args)
        
        #print(argvalues)        
        #print(self.nargs)
        
        if len(argvalues) == self.nargs:
            
            return  self.funct(*argvalues)
        
        elif len(argvalues) > self.nargs:
            raise TypeError("Wrong number of arguments")
        
        else:
            newfunctor = CurriedFunction(self.funct, argvalues)
            return newfunctor            
        
def curry(function):
    return CurriedFunction(function)


to_list   =  list     # Convert generator to list
to_tuple  =  tuple    # Convert gernerator to tuple
to_stream =  iter     # Convert iterable object to generator

reverse = reversed


@curry
def add(x, y):
    return x + y
@curry
def sub(x, y):
    return y - x

@curry
def div(x, y):
    return y/x

@curry
def mul(x, y):
    return x*y

@curry
def pow(x, y):
    return x**y


# curry :: ((a, b) -> c) -> a -> b -> c

def identity(x):
    return x
    
def const(constant):    
    return lambda *args, **kwargs: constant

def curry2(func):
    return lambda a, b: f((a, b))

def curry3(func):
    return lambda a, b, c: func((a, b, c))

def uncurry2(func):
    """     uncurry :: (a -> b -> c) -> (a, b) -> c  """
    return lambda tpl: func(tpl[0], tpl[1])

def uncurry3(func): 
    return lambda tpl3: func(tpl3[0], tpl3[1], tpl3[2])    
    
flip = lambda f: lambda x, y: f(y, x)

def zipWith(func, stream1, stream2):
    return map(lambda e: func(e[0], e[1]), zip(stream1, stream2))

def take(n, iterable):
    
    for i in range(n):
        
        try:
            yield next(iterable)            
        except:
            pass


def foldr(function, initial, iterable):
    return reduce(function, iterable, initial)
       
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

def root(a):
        
    f = lambda x: 0.5*(a/x + x)    
    return last(take(10, iterate(f, 1.0)))


def pairs(alist):
    return zip(alist, tail(iter(alist)))

def lagdiff(alist):
    ialist = iter(alist)
    return to_list(zipWith (lambda x, y: y-x,  ialist, tail(ialist)))

def growth(alist): 
    ialist = iter(alist)
    return to_list(zipWith (lambda x, y: (y-x)/x,  ialist, tail(ialist)))



def infinite(start=0):
    idx = start

    while True:
        yield idx
        idx +=1

def infinite_alternate(start=0):
    idx = start
    sig = 1

    while True:
        yield idx*sig
        idx +=1
        sig *= -1

is_even = lambda n: n % 2 == 0
is_odd  = lambda n: n % 2 == 1


def allmap(predicate, iterable):    
    return all(map(predicate, iterable))
    
def anymap(predicate, iterable):    
    return any(map(predicate, iterable))


def mainf(function):
    """
    Decorator to set the function
    as the main function.
    
    Example:
    
    @mainf
    def myfunction():
        print("Hello World")
    
    """
    
    if __name__ == "__main__" : 
        function()


def profile(function):
    """
    Measuere the execution time.
    of a function.
    """
    
    def _ (*args, **kwargs):
        tstart = time.time()
        result = function(*args, **kwargs)
        tend   = time.time()
        
        print("Time taken: ", (tend - tstart))
        return result
    
    return _
    


@curry
def call_with(arguments, function):  
    """
    Example:
        
        >>> def f(x, y): return x*y - 10*x/y
        >>> call_with((2, 4))
        <__main__.CurriedFunction at 0xb254f0ac>
        
        >>> call1 = call_with((2, 4))

        >>> call2 = call_with((6, 8))

        >>> call1(f)
        3.00000

        >>> call2(f)
        40.50000

        >>> list(map(call1, (f, f2)))
        [3.00000, 6]

        >>> list(map(call2, (f, f2)))
        [40.50000, 14]
            
    """
    return function(*arguments)
    


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
def nth(n, stream):
    """

    nth(N, List) -> Elem

    > lists:nth(3, [a, b, c, d, e]).
    c

    Idea from: http://erldocs.com/17.3/stdlib/lists.html
    """
    i = 0
    while i < n:
        next(stream) # Consume stream
        i += 1
    return next(stream)
    
        
        

#---------------------------------------------------#

def until_converge(stream, eps, itmax):
    
    a = next(stream)
    b = next(stream)
    
    n = 1

    while True:
        
        if n > itmax:
            break
        
        if abs((b-a)/a) < eps:
            break

        yield a
        yield b

        a = b
        b = next(stream)
        n = n + 1
        
    print("n = ", n)

def summation(stream):
    
    s = 0
    while True:       
        s += next(stream)
        yield s
    
def aitken(stream):
    Sn  = next(stream)
    Sn1 = next(stream)
    Sn2 = next(stream)

    
    while True:
    
        s = Sn2 - (Sn2 - Sn1)**2 / (Sn2 - 2*Sn1 + Sn)
        Sn, Sn1, Sn2 = Sn1, Sn2, next(stream)
        yield s
        
def infinite_serie(term, inext, start):
    
    i = start
    
    while True:
        yield term(i)
        i = inext(i)
        

def fixed_point(iterator, eps, itmax, guess):

    stream = iterate(iterator, guess)   
    return last(until_converge(stream, eps, itmax))
    

def nsolver(f, df, guess, eps, itmax):
    
    iterator = lambda x: x - f(x)/df(x)   
    return fixed_point(iterator, eps, itmax, guess)
    

def pi_serie():
    
    s = 0
    i = 1
    k = 0
    
    while True:
        
        s = s + i*4/(2*k+1)
        i = -1 * i
        k = k + 1
        yield s
        

def fib(n):
    
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
