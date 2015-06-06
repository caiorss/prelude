#---------------------------------#
#   STRINGS                       #
#---------------------------------#

from .prelude import curry

import re

@curry
def join(strmid, stringsList):
    return strmid.join(stringsList)
    
@curry
def replace(oldValue, newValue, string):    
    return string.replace(oldValue, newValue)

@curry
def replaceList(alistOld, new, string):
    """
    :param alistOld:
    :param new:
    :param string:
    :return:

    >>> import prelude as p
    >>> from   prelude import Str as st
    >>> f = st.replaceList("/&*.$", "")
    >>>
    >>> f("he&*ll.o wo/$&rld")
    'hello world'
    >>>
    >>> st.replaceList("/&*.$", "", "he&*ll.o wo/$&rld")
    'hello world'
    >>>
    >>> p.mapl(st.replaceList("/&*.$", ""), ["he&*ll.o wo/$&rld", "s....om$*e"])
    ['hello world', 'some']
    >>>
    """

    newstr = string

    for i in alistOld:
        newstr = newstr.replace(i, new)
    return newstr

@curry
def split(separator, string):
    return string.split(separator)


def splitLines(string):
    """
    splitLines :: str -> [str]

    :param string:
    :return:
    """
    return string.splitlines()

def joinLines(lines):
    """
    splitLines :: [str] -> str

    :param lines:
    :return:
    """
    return str.join("\n", lines)

@curry    
def strip(chars, string):
    return string.strip(chars)


@curry
def hasPrefix(prefix, string):
    return string.startswith(prefix)

@curry
def hasSuffix(suffix, string):
    return string.startswith(suffix)


@curry
def stripPrefix(prefix, string):
    """
    Remove prefix from a string.

    :signature:     stripPrefix :: -> str -> str -> str
    :param prefix:  Prefix to be removed from a String
    :param string:  String
    :return:        String Without the prefix


    Example:

    >>> from prelude import Str as Str
    >>> import prelude as p
    >>> p.mapl(Str.stripPrefix("http://"), [ "http://www.google.co.uk", "http://www.yahoo.co.uk", "http://www.youtube.com"])
    ['www.google.co.uk', 'www.yahoo.co.uk', 'www.youtube.com']
    """

    if string.startswith(prefix):
        return string[len(prefix):]

    return string


@curry
def stripSuffix(suffix, string):
    """
    Remove suffix from a string.

    :signature:     stripSuffix :: -> str -> str -> str

    :param suffix:
    :param string:
    :return:

    >>> import prelude as p
    >>> from prelude import Str as Str
    >>> p.mapl(Str.stripSuffix(".md"), [ "README.md", "Topic1.md", "Topic2.md", "Topic3"])
    ['README', 'Topic1', 'Topic2', 'Topic3']
    """

    if string.endswith(suffix):
        return string[:-len(suffix)]

    return string


@curry
def addSuffix(suffix, string):
    return string + suffix

@curry
def addPrefix(prefix, string):
    return prefix + string




#substr = re.sub()
