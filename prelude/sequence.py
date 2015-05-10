from .prelude import curry, last

def sign(x):
    
    if    x>0: return  1
    elif  x<0: return -1
    else: return 0
        

def infinite(start=0):
    count = start
    while True:
        yield count
        count += 1

def serieGen(elemFunction, startIdx=0, startElem=None):
    """
    :param start: Start value of the serie
    :param nexElement: Function f such that - em+1 = f(em)
    :return: Generator to the serie e0, e1, e2, ...
    :rtype:  generator
    
    e0 = start
    e1 = f(e0)
    e2 = f(e1)
    ...
    
    """
    idx = startIdx
    elem = startElem
    
    
    if startElem is not None:
        
        yield elem
              
        while True:              
            yield elem
            elem = elemFunction(idx, elem)           
            idx += 1
    
    else:
        while True:                          
            elem = elemFunction(idx, elem)           
            yield elem
            idx += 1

def serieLaggedGen(start, nextElement):
    """
    
    :param start: Tuple of start values (e0, e1, e2, .. em)
    :param nextElement: Function em+1 = f(e0, e1, e2, ... em)
    :return: Generator of sequence:  e0, e1, e2, ... em, em+1, em+2 ...
    :rtype: generator
    
    The next element is defined as function of the m old values
    
    Example: Fibbonaci Serie
    
    >>> takel(10, summation(serieLaggedGen((1, 1), lambda x, y: x+y)))
    [1, 2, 4, 7, 12, 20, 33, 54, 88, 143]

    
    """
   
    memory = start   
    rotate = lambda atuple, element:  atuple[1:] + (element,)
    
    for e in memory:
        #print("e = ", e)
        yield e       
    
    #print("memory = ", memory)
    
    #enext = nextElement(*memory)
    #yield enext
    
    while True:
        
        enext  = nextElement(memory)
        memory = rotate(memory, enext) 
        #print(enext)
        #print(memory)
        yield enext
        

def serieFilter(serie, filterFunction, nargs):
    
    rotate = lambda atuple, element:  atuple[1:] + (element,)
    
    args = tuple(next(serie) for i in range(nargs))
    
    #print("args = ", args)
    
    #print("---")
    
    while True:
        
        try:            
            enext  = filterFunction(args)
            yield enext
            args = rotate(args, next(serie))                                     
            #print("args = ", args)            
            
        except StopIteration:
            break
                          
    
def accum(serieGenerator):
    """
    Accumulated List of Old Values
    """
    
    acc = [next(serieGenerator)]   
    yield acc.copy()
    
    while True:        
        acc.append( next(serieGenerator))
                        
        #print((id(acc))
        #print("acc = " + str(acc))
        
        yield acc.copy()
        
        
def summation(stream):
    """
    Acummulated Sum Serie    
    """

    acc = 0
    while True:
        acc += next(stream)
        yield acc
 

def converge(eps, itmax, stream, debug=False):
    """
    Take a serie until converge
    
    """
    
    itmax_ = itmax // 2
    
    a = next(stream)
    b = next(stream)

    n = 1

    while True:

        if n > itmax_:
            break
        
        if    a == 0 and abs(b) < eps: break
        elif abs((b-a)/a) < eps: break

        yield a
        yield b

        a = b
        b = next(stream)
        n = n + 1
    
    err = abs((b-a)/a)
    
    if debug:
        print("iterations: ", 2*n)
        print("Error:      ", err)
        
    if err > eps:
        raise Exception("Serie don't converge: error > eps")
        
    #print("n = ", n)

def iterate(iterator, x0):
    """
    Fixed Point Solver
    
    :param iterator: Function β(x) --> Xn+1 = β(Xn)
    :param x0:       Initial guess
    :return:         Generator [x0, x1, x2, ... ] Xn+1 = β(Xn)
      
    """
    #return last(converge(stream, eps, itmax, debug=debug))
    y = x0

    while True:
        
        yield y
        y = iterator(y)
        

def fixedPoint(eps, itmax, x0, function, debug=False):   
    return last(converge(eps, itmax, iterate(function, x0), debug=debug))


def aitken(stream):
    Sn  = next(stream)
    Sn1 = next(stream)
    Sn2 = next(stream)


    while True:

        s = Sn2 - (Sn2 - Sn1)**2 / (Sn2 - 2*Sn1 + Sn)
        Sn, Sn1, Sn2 = Sn1, Sn2, next(stream)
        yield s
