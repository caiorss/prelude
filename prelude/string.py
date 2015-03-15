#---------------------------------#
#   STRINGS                       #
#---------------------------------#

from .prelude import curry

def joinstr(stringsList):    
    return "".join(stringsList)
    
@curry
def replace(oldValue, newValue, string):    
    return string.replace(oldValue, newValue)

@curry
def split(separator, string):
    return string.split(separator)
    
@curry    
def strip(chars, string):
    return string.strip(chars)


@curry
def has_prefix(prefix, string):
    return string.startswith(prefix)

@curry
def has_suffix(suffix, string):
    return string.startswith(suffix)
