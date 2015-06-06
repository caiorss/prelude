# PRELUDE

Prelude is a functional programming library inspired on FP languages like Haskell that has nice features like fast lambda 
functions, easy currying, curried functions and many useful Higher Order Functions



## Predicate Functions

### Type Checking Functions

```
is_tuple    Test if  is a tuple
is_list     Test if os a list
is_gen      Test if is a generator
is_str      Test if is a string

is_function Test if is a function (has __call__ method)


```


## Operator Functions

```
              Argument    Result
add             x y         x+y
sub             y x         y-x
div             y x         y/x
mul             x y         x*y
pow             x y         x^y
contains       x alist     x in alist
```

Examples:

```python


>>> from prelude import mapf, mapl, add, sub, div, mul, pow, contains
>>>
>>>
>>> add(20, 23)
43
>>> add(20)(30)
50
>>> 
>>> mapl(add(20), range(10))
[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
>>> 
>>> mapl(sub(100), range(10))
[100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
>>> 

>>> 
>>> add20toList = mapl(add(20))
>>> 
>>> add20toList(range(10))
[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
>>> 
>>> add20toList(range(5))
[20, 21, 22, 23, 24]
>>> 
>>> 

```

## Currying



```python
>>> from prelude import mapf, mapl, curry, splitstr, joinstr
>>> 
>>> def f(x, y, z): return x**2 + y**2 - 10*z
... 
>>> f = curry(f)
>>> f(3, 5, 6)
-26
>>> f(3)(5)(6)
-26
>>> f(3)(5, 6)
-26
>>> f35 = f(3, 5)
>>> f35(6)
-26
>>> f35(10)
-66
>>> f35(1)
24
>>> 
>>> mapl(f35, [-1, -2, 3, 4])
[44, 54, 4, -6]
>>> 
>>> f35map = mapl(f35)
>>> f35map([-1, -2, 3, 4])
[44, 54, 4, -6]
>>>
>>>
>>> f5 = f(5)
>>> f5(5, 6)
-10
>>> f5(5, -6)
110
>>> 
>>> mapl(uncurry(f5))([(3, 4), (5, 6), (7, 8)])
[-6, -10, -6]
>>> 
>>> 
>>> 
>>> f5lst = mapl(uncurry(f5))
>>> f5lst ([(3, 4), (5, 6), (7, 8)])
[-6, -10, -6]
>>> 



>>> 
>>> joinlines = joinstr("\n")
>>> joinlines(["line1", "line2", "line3"])
'line1\nline2\nline3'
>>> 

```

## Higher Order Functions

```python

>>> from prelude import nth,  mapf, mapl, curry, splitstr, slice
>>> 
>>> mapl(nth(1), ["hello", "world", "character"])
['e', 'o', 'h']
>>> 
>>> 
>>> mapl(slice(2, 4), ["hello23", "world23", "character"])
['ll', 'rl', 'ar']
>>> mapl(slice(2, 5), ["hello23", "world23", "character"])
['llo', 'rld', 'ara']
>>> 
>>> 

```

## Piping

The pipilining object Stream() used with the operator >> is similar to F# operator |>. Example:

```python
>>> from prelude import to_value, to_list, Stream, mapl, mapf, add, mul
>>> 
>>> Stream() >> range(8) >>  mapf(add(10)) >> mapf(mul(8.5)) >> to_list 
[85.0, 93.5, 102.0, 110.5, 119.0, 127.5, 136.0, 144.5]
>>> 
>>> 
>>> Stream(range(8)) >>  mapf(lambda x: x**2) >> mapf(lambda y: y/10) >> to_list 
[0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9]
>>> 
>>> Stream(10)  >> (lambda x: x/10) >> (lambda x: x*8) >> to_value 
8.0
>>> 
>>> 
```

## Lambda Operator

The lambda operator is syntax sugar to create fast lambda functions.

```python

>>> from prelude import X
>>> 
>>> 
>>> (X - 10)(20)
10
>>> 
>>> (lambda x: x-10)(20)
10
>>> 
>>> f_lst = mapl(X/10)
>>> f_lst(range(8))
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
>>> 
>>> 

```


## Modules

The submodule prelude.Modules implements a OCaml like module system.

Example:

```python
>>> from prelude.Modules import (Fp, Obj, Op, List, Chain)

>>> 

>>> Chain([(10, 20), (40, 60), (80, 100), (20, 30)]) >> List.mapl(Fp.uncurry(Op.add)) >> sum
    Chain : 360

>>> 

>>> Chain([(10, 20), (40, 60), (80, 100), (20, 30)]) \
 ...    >> List.mapl(Fp.uncurry(Op.add)) \
 ...    >> List.mapl(Op.mul(2)) \
 ...    >> sum
    Chain : 720

>>> 

>>> ( Chain([(10, 20), (40, 60), (80, 100), (20, 30)])
 ...    .b(List.mapl(Fp.uncurry(Op.add)))
 ...    .b(sum)
 ...)
    Chain : 360

>>> 
```

## Decorators

