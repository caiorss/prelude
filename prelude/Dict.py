from .monad import Nothing, Just
from .prelude import curry

#####################################
# DICTIONARY                        #
#####################################

def dictzip(keys, values):
    """
    >>> dictzip(["x", "y", "z"], [1, 2,3])
    {'y': 2, 'x': 1, 'z': 3}
    """
    return dict(list(zip(keys, values)))


def transpose(matrix):
    return list(zip(*matrix))


def mapdict_values(function, dic):
    """
    Apply a function to a dictionary values,
    creating a new dictionary with the same keys
    and new values created by applying the function
    to the old ones.

    :param function: A function that takes the dictionary value as argument
    :param dic:      A dictionary
    :return:         A new dicitonary with same keys and values changed

    Example:

    >>> dic1 = { 'a' : 10, 'b' : 20, 'c' : 30 }
    >>> mapdict_values(lambda x: x*2, dic1)
    {'a': 20, 'b': 40, 'c': 60}
    >>> dic1
    {'a': 10, 'b': 20, 'c': 30}
    """
    return dict(map(lambda x: (x[0], function(x[1])), dic.items()))


def mapdict_keys(function, dic):
    """
    Apply a function to a dictionary keys,
    creating a new dictionary with the same values
    and new values created by applying the function
    to the old ones.

    :param function: A function that takes the dictionary key as argument
                      and returns a new dictionary key
    :param dic:      A dictionary
    :return:         A new dicitonary with same keys and values changed

    Example:

    >>> dic1 = { 'a' : 10, 'b' : 20, 'c' : 30 }
    >>>
    >>> mapdict_keys(lambda x: str(x) + "_hello", dic1)
    {'a_hello': 10, 'b_hello': 20, 'c_hello': 30}
    >>>
    >>> dic1
    {'a': 10, 'b': 20, 'c': 30}
    """
    return dict(map(lambda x: (function(x[0]), x[1]), dic.items()))


def merge_dict(*dicts):
    """
    Merge a list of dictionaries returning a new one.
    
    :param dictionaries: A list of dictionaries
    :return:             A new dicitonary
    :rtype:              dict
    """
    dic = {}
    for d in dicts:
        if d is not None:
            dic.update(d)
    return dic


def reverse_dict(dic):
    """
    
    """
    return dictzip(*reverse(unzip_l(dic.items())))


def get(property):
    """
    >>> user =  {'name': 'Bemmu', 'uid': '297200003'}
    >>> get("name")(user)
    'Bemmu'
    """

    def get_property(object):

        if is_dict(object):
            return object.get(property)
        else:
            return getattr(object, property)

    return get_property


def pluck(property):
    """

    This pattern of combining splat and get is very
    frequent in JavaScript code.
    So much so, that we can take it up another level:

    :param property:
    :return:
    
    Example:
    
   users = [
       {
       "name" : "Bemmu",
       "uid" : "297200003"
       },
       {
       "name" : "Zuck",
       "uid" : "4"
       }
   ]   
   >>> get_name = pluck("name")
   >>> get_name(users)
   ['Bemmu', 'Zuck']

    
    
    
    """

    def fun(object_array):


        if is_list(object_array):

            if is_dict(object_array[0]):
                out = map(lambda obj: obj.get(property), object_array)
            else:
                out = map(lambda obj: getattr(obj, property), object_array)
        else:

            if is_dict(object_array):
                out = object_array.get(property)
            else:
                out = getattr(object_array, property)

        return list(out)

    return fun


@curry
def bind_maybe(ma, f_a_to_mb):
    return f_a_to_mb(ma.from_just)


@curry
def lookup(key, dic):
    if not isinstance(dic, dict):
        return Nothing

    if dic.get(key):
        return Just(dic.get(key))
    else:
        return Nothing


@curry
def lookup_nested(keys, dic):
    ikeys = iter(keys)

    value = lookup(next(ikeys), dic)

    for key in ikeys:
        value = value.bind(lookup(key))
    return value
    
