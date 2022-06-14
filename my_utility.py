# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  

# Initialize weights
def iniWs():    
    pass

# Initialize weights for one-layer    
def iniW(prev, next):
    r = np.sqrt(6 / (next + prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of SNN
def forward(x, W):
    act = []
    # Activacion para las capas de entrada y ocultas
    for i in range(len(W)-1):
        act.append(act_function(np.dot(W[i], x)))
    
    act.append(act_sigmoid(np.dot(W[-1], act[-1]))) # Activacion de la capa de salida
    return(act) 

# Activacion para la capa de salida
def act_sigmoid(x):
    return(1/(1+np.exp(-x)))

# Funcion de activacion: Tangente hiperbolica
def act_function(x):
    return(np.tanh(x))
  
# Derivada de la funcion Tangente hiperbolica
def deriva_act(x):
    return(1/(1+np.abs(x)))
    
   
#Feed-Backward of SNN
def gradW(a, x, W, V):
    gW = []
    costo = 0
    # Gradiente de la capa de salida
    for i in range(len(W)-1):
        

    return()    

# Update Ws
def updW():
    return

# Measure
def metricas(x,y):
    return()
    
#Confusion matrix 
def confusion_matrix(z,y):
    return(cm)
#-----------------------------------------------------------------------