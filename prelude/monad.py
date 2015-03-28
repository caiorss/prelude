"""
    https://github.com/ericmoritz/fp/blob/0.1.2/fp/monads/maybe.py
    http://javcasas.github.io/posts/2014/Jul/17/python-monads/
    
    http://blog.osteele.com/posts/2007/12/cheap-monads/
    
    
 data = {"foo": {"bar": {"baz": "bing"}}}
 
  >>> data = {"foo": {"bar": {"baz": "bing"}}}
>>> data['foo']['bong']['baz']
Traceback (most recent call last):
...
KeyError: 'bong'
This is a very common problem when processing JSON. Often, with
JSON, a missing key is the same as null but in Python a missing
key raises an error.
`dict.get` is often the solution for missing keys but is
still not enough:
 
 get_nested(Maybe, data, "foo", "bong", "baz")



data = {"foo": {"bar": {"baz": "bing"}}}
test_1 = lookup("foo", data).bind(lookup("bong")).bind(lookup("baz")) == Nothing
test_2 = lookup_nested(["foo", "bong", "baz"], data) == Nothing
test_3 = lookup_nested(["foo", "bar", "baz"], data)  == Just("bing")

>>> lookup("foo", data) >> lookup("bong") >> lookup("baz")
    Nothing

>>> lookup("foo", data) >> lookup("bar") >> lookup("baz")
    Just('bing')


"""

from .prelude import curry, profile, flip, reduce, foldr

class Maybe(object):
    
    def __init__(self, value):
        self.__value = value
    
    def __eq__(self, other):
        return self.__value == other.__value

    def __ne__(self, other):
        return self.__value != other.__value
    
    def __repr__(self):
        if self.is_just:
            return "Just({0!r})".format(self.__value)
        else:
            return "Nothing"
            
    def __add__(self, other):
        
        if other.is_nothing or self.is_nothing:
            return Nothing        
        else:
            return Just(self.__value + other.__value)
    
    def __sub__(self, other):
        
        if other.is_nothing or self.is_nothing:
            return Nothing
        else:
            return Just(self.__value - other.__value)

    def __rsub__(self, other):
        
        if other.is_nothing or self.is_nothing:
            return Nothing
        else:
            return Just(other.__value - self.__value)

    
    def __mul__(self, other):
        
        if other.is_nothing or self.is_nothing:
            return Nothing
        else:
            return Just(self.__value * other.__value)
    
    def __div__(self, other):
        
        if other.is_nothing or self.is_nothing:
            return Nothing
        else:
            return Just(self.__value / other.__value)
    
    
    @property
    def is_just(self):
        return self.__value is not None
    
    @property
    def is_nothing(self):
        return self.__value is None
    
    def from_just(self):
        if self.is_just:
            return self.__value
        else:
            raise ValueError("Called on Nothing")     
    
    @classmethod
    def lift(cls, f):
        """
        liftM :: Monad m => (a1 -> r) -> m a1 -> m r
        let f x = 2*x - 10
        
            λ> liftM f (Just 1)
            Just (-8)
            λ> 
            λ> liftM f (Nothing)
            Nothing
            λ> 
        """
        
        def _(maybe):
            
            if maybe.is_nothing:
                return Nothing
            
            y = f(maybe.__value)
            if y is None:
                return Nothing
            else:
                return Just(y)
        
        return _
    
    #@classmethod
    def bind(self, f_a_to_mb):
        """
        (>>=) :: Monad m => m a -> (a -> m b) -> m b

        """
        return f_a_to_mb(self.__value)
    
    def __rshift__(self, f_a_to_mb):
        """
        Bind operator
        
        (>>=) :: Monad m => m a -> (a -> m b) -> m b
        """
        if self.is_nothing:
            return Nothing
        return f_a_to_mb(self.__value)
    
Just = Maybe
Nothing = Maybe(None)








