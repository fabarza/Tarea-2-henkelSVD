# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  

# Initialize weights
def iniWs():    
    ...
    return()

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward():
    ...    
    return() 

#Activation function
def act_function(x):
    ''' Funcion tangente hiperbolica 
    INPUT: valor de entrada para la funcion tangente hiperbolica
    OUTPUT: valor de la funcion tangente hiperbolica
    '''
    if x == None:
        print("x value is None")
    else:
        tg_h = (exp(x)-exp(-x)) / (exp(x)+exp(x)) 
        return tg_h
  
# Derivate of the activation funciton
def deriva_act(x):
    ''' Derivada de funcion tangente hiperbolica
    INPUT: valor de entrada para la derivada funcion tangente hiperbolica
    OUTPUT: valor de la derivada de funcion tangente hiperbolica
    '''
    if x == None:
        print("x value is None")
    else:
        tg_h_dv = 1 / ((np.cosh(x))**2) 
        return tg_h_dv
   
#Feed-Backward of SNN
def gradW(...):    
    ...    
    return()    

# Update Ws
def updW(...):
    ...    
    return(...)

# Measure
def metricas(x,y):
    return()
    
#Confusion matrix for
def confusion_matrix(z,y):
    ...    
    return(cm)
#-----------------------------------------------------------------------