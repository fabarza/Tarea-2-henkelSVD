# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  

#Inicializacion de pesos 
def iniWs(x,y,param_snn):
    # Asignacion de parametros
    next = param_snn[3]
    prev = x.shape[0]
    # Inicializacion de pesos
    W = [iniW(next, prev)]
    V = [np.zeros(W[0].shape)]

    # Inicializacion de pesos de la capa de entrada
    for i in range(4, len(param_snn)):
        prev = next
        next = param_snn[i]
        W.append(iniW(next, prev))  # Guardamos los pesos por capa
        V.append(np.zeros(W[i-3].shape)) # Velocidad

    W.append(iniW(y.shape[0], W[-1].shape[0])) # Pesos de la capa de salida
    V.append(np.zeros(W[-1].shape)) # Velocidad de salida

    return (W, V)

# Initialize weights for one-layer    
def iniW(next, prev):
    r = np.sqrt(6 / (next + prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r
    return(w)

# Feed-forward of SNN
def forward(x, W):
    act = []
    z = np.matmul(W[0], x)
    act.append(act_function(z))

    # Activacion para las capas de entrada y ocultas
    for i in range(1, len(W)-1):
        z = np.matmul(W[i], act[i-1])
        act.append(act_function(z))
    
    act.append(act_sigmoid(np.matmul(W[-1], act[-1]))) # Activacion de la capa de salida

    return(act) 

# Activacion para la capa de salida
def act_sigmoid(x):
    return(1/(1+np.exp(-x)))

# Derivada de la funcion de activacion sigmoidea
def deriva_act_sigmoid(x):
    return(act_sigmoid(x) * (1 - act_sigmoid(x)))

# Funcion de activacion: Tangente hiperbolica
def act_function(x):
    mat = np.tanh(x)
    return(mat)
  
# Derivada de la funcion Tangente hiperbolica
def deriva_act(x):
    return(1/(1+np.abs(x)))
    
   
#Feed-Backward of SNN
def gradW(a, x, y, W):
    gW = []
    # Gradiente de la capa de salida
    z = np.dot(W[-1], a[-2]) # z_l = a_{l-1} * W_{l}
    err = (a[-1] - y)        # error = a_{l} - y
    delta = err * deriva_act_sigmoid(z) # delta_l = error * derivada de la funcion de activacion
    
    costo = mse(a[-1], y) # Costo de la capa de salida
    
    gW.append(np.dot(delta, a[-2].T)) # Gradiente de la capa de salida
    
    # Gradiente de las capas ocultas
    for i in reversed(range(1, len(W)-1)):
        lSide = np.dot(W[i+1].T,delta)
        z = np.dot(W[i], a[i-1])
        rSide = deriva_act(z)
        delta = lSide * rSide
        gW.insert(0, np.dot(delta, a[i-1].T))

    # Gradiente de la capa de entrada
    lSide = np.dot(W[1].T,delta)
    z = np.dot(W[0], x)
    rSide = deriva_act(z)
    lSide = lSide * rSide
    gW.insert(0,np.dot(lSide, x.T))

    return(gW, costo)    

# Mean squared error
def mse(z,y):
    lSide = np.divide(1, z.shape[1]*2)
    rSide = np.power(z - y, 2)
    rSide = np.sum(rSide)
    cost = lSide * rSide
    return(cost)


# Update Ws
def updW(gW, W, V, lRate):
    epsilon = 10**(-9)
    beta = 0.9

    for i in range(len(W)):
        V[i] = (beta * V[i]) + (1 - beta) * np.power(gW[i], 2) # Actualizacion de la velocidad
        den = np.sqrt(V[i] + epsilon)
        gRMS = (1/(den)) * gW[i]
        W[i] = W[i] - (lRate * gRMS) # Actualizacion de los pesos

    return (W, V)

# Measure
def metricas(x,y):
    cm = confusion_matrix(x,y)
    tp = np.diag(cm)
    tn = np.sum(cm) - tp
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    fscore = 2 * tp_rate * tn_rate / (tp_rate + tn_rate)
    # Promedio del fscore
    mFscore = np.mean(fscore)
    fscore = np.append(fscore, mFscore)

    return(cm, fscore)
    
#Confusion matrix 
def confusion_matrix(z,y):
    clases = np.arange(0, z.shape[0], 1)
    d = y.shape[0]
    conf_m = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            conf_m[i, j] = np.sum((np.argmax(y, axis=0) == clases[i]) & (
                np.argmax(z, axis=0) == clases[j]))

    return(conf_m)
#-----------------------------------------------------------------------