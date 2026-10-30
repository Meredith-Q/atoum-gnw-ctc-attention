###################################################################
###################################################################
## 
## Generic functions used in the Notebook 
## 
## 


###################################################################
###################################################################
## IMPORT SECTION 
## 

import numpy as np



###################################################################
###################################################################
## FUNCTIONS 
## 

###################################################################
## foo must be a boolean function 
## find_bounds will then find the bounds in wich the function is True
## (This assumes that Domain of Truth is a Segment)

def find_bounds(foo,x_bounds,y_bounds,eps,step):

    xx = np.arange(*x_bounds,step)
    result = np.zeros((len(xx)))

    for i in range(len(xx)):
        y_min = y_bounds[0]
        y_max = y_bounds[1]
        d = 1
        while d > eps:

            y_new = (y_min+y_max)/2
            f = foo(xx[i],y_new)
            if f:
                y_max = y_new
            else:
                y_min = y_new
            d /= 2
        result[i]=y_max

    return result
