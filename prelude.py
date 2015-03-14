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

to_list   =  list     # Convert generator to list
to_tuple  =  tuple    # Convert gernerator to tuple
to_stream =  iter     # Convert iterable object to generator


reverse = reversed

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
    

class List(object):
    """
    List Monad
    
    """
    
    def __init__(self, value):
        self.value = value
        
    def bind(self, function):
        return List(function(self.value))
        
    def filter(self, predicate):
        return List(filter(predicate, self.value))
    
    def reject(self, predicate):
        return List(filter(lambda x: not predicate(x), self.value))
    
    def fold(self, initial_value, function):
        return List(reduce(function, self.value, initial_value))
    
    
    def get(self, attribute):
        return self.map(lambda x: getattr(x, attribute))
    
    def getkey(self, key):
        return self.map(lambda x: x.git(key))
    
    def to_list(self):
        return list(self.value)
    
    def to_tuple(self):
        return tuple(self.value)
        
    def iter(self):
        return List(iter(self.value))
    
    def reverse(self):
        return List(reversed(self.value))
        
    def map(self, function):
        return List(map(function, self.value))
    
    
    def tail(self):
        return List(tail(self.value))
    
    def last(self):
        return last(self.value)
    
    def head(self):
        return head(self.value)
    
    def zip(self, sequence):
        return List(zip(self.value, sequence))
    
    def zipwith(self, function, sequence):
        return List(zipWith(function, self.value, sequence))
        
    def take(self, n):
        return List(iter(take(n, self.value)))
        
    def takewhile(self, predicate):
        return List(takeWhile(predicate, self.value))
        
    def pairs(self):
        return List(pairs(self.value))
    
    def column(self, n):
        return self.map(lambda x: x[n])
        
    def foreach(self, function):
        foreach(function, self.value)

f = lambda a: lambda x: 0.5*(a/x + x)    
    
        

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
