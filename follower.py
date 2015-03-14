"""
Implementation of unix: tail -f
"""

import time

def foreach(function, iterable):
    for element in iterable:
        function(next(iterable))


def watchfile(afile):
    afile.seek(0, 2) # End of file
    
    while True:
        line = afile.readline()
        
        if not line:
            time.sleep(0.1)
            continue
        
        yield line

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

        
        
    
@mainf
def main():
    
    print ("Main function")
    
    logfile = open("/var/log/syslog")
    foreach(print, watchfile(logfile))
