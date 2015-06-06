"""
Functions grouped in modules like OCAML modules. It makes easier to find
and organize functions. 


Examples:

>>> from prelude.Mods import (Fp, Obj, Op, List)

>>> class Point:
 ...    def __init__(self, x, y): 
 ...        self.x = x
 ...        self.y = y
 ...   
 
>>> points = [Point(1, 2), Point(2.23, 10), Point(10, 20)]


>>> List.mapl(Obj.attrib("x"), points)
    [1, 2.23000, 10]

>>> List.mapl(Obj.attrib("y"), points)
    [2, 10, 20]

>>> get_x = List.mapl(Obj.attrib("x"))
>>> get_y = List.mapl(Obj.attrib("y"))

>>> get_x(points)
    [1, 2.23000, 10]

>>> get_y(points)
    [2, 10, 20]

"""

from . import prelude as p
from . import Str
from . import Dic
from . import Chain

__all__ = [ "Str", "List", "Obj", "Op", "Type", "Chain"]

class Op(object):
    """
    Operators curried functions Module
    
    Functions:
    
        
    .add    +
    .sub    -
    .mul    *
    .div    /
    .pow    **  
    .le     <=  Less or equal than
    .lt     <   Less Than
    .gt     >   Greater
    .ge     >=  Greater or Equal
    .eq     ==  Equal
    .neq    !=  Not Equal
    """

Op.add = p.add
Op.sub = p.sub
Op.mul = p.mul
Op.div = p.div
Op.pow = p.pow
Op.le  = p.le
Op.lt  = p.lt 
Op.ge  = p.ge
Op.gt  = p.gt
Op.eq  = p.eq
Op.neq = p.neq


    
class Type(object):
    """
    Type check functions
    
    Functions in this module:

        .is_num 
        .is_int 
        .is_float 
        .is_dict 
        .is_str 
        .is_function 
        .is_empty 
        .is_gen 
        .is_none 
        .is_tuple 
        .is_list 
    """

Type.is_num = p.is_num
Type.is_int = p.is_int
Type.is_float = p.is_float
Type.is_dict = p.is_dict
Type.is_str = p.is_str
Type.is_function = p.is_function
Type.is_empty = p.is_empty
Type.is_gen = p.is_gen
Type.is_none = p.is_none
Type.is_tuple = p.is_tuple
Type.is_list = p.is_list


class List(object):
    """
    Functions to list manipulation.
    
    """
   
List.mapl       = p.mapl
List.map        = p.mapf
List.filterl    = p.filterl
List.filter     = p.filterf
List.foldr      = p.foldr
List.foldl      = p.foldl
List.foldl1     = p.foldl1
List.unique     = p.unique
List.tail       = p.tail
List.nth        = p.nth
List.zip        = zip
List.zipl       = p.zipl

List.contains   = p.contains


class Obj(object):
    """
    Functions to Manipulate objects and classes
    
    .mcall      Call an object method
    """

Obj.mcall  = p.mcall
Obj.mcallm = p.mcallm
Obj.attrib = p.attrib
Obj.get    = Dic.get

class Fp(object):
    """
    General functions
    
    """

Fp.curry   = p.curry
Fp.uncurry = p.uncurry
Fp.cpipe   = p.cpipe
Fp.compose = p.compose
Fp.juxt    = p.juxt
Fp.identity = p.identity
Fp.constant = p.constant

