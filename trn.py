# SNN's Training :

import pandas     as pd
import numpy      as np
import my_utility as ut


#Inicializacion de pesos 
def iniWs():
    pass

#Save weights of the SNN
def save_w():
    pass 
    
#SNN's Training 
def snn_train(x,y,param):    
    W,V,S = iniWs()
    Costo = []
    for iter in range(MaxIter):        
        a       = ut.forward(...)
        gW,cost = ut.gradW(...)        
        W,V,S   = ut.updW(...)
        Costo.append(Cost)
    return(W,Costo)

# Load data from xData.csv, yData,csv
def load_data_trn(xData,yData):
    rng = np.random.default_rng() # Generador de numeros aleatorios para shuffle

    # Lectura de datos
    xe = np.genfromtxt(xData, delimiter=",", dtype=np.float64)
    ye = np.genfromtxt(yData, delimiter=",", dtype=np.float64)
    
    xe = np.append(xe, ye, axis=1) # Union de las caracteristicas con las etiquetas

    rng.shuffle(xe) # Desordenar las filas
    
    # Separo las caracteristicas de las etiquetas
    xe, ye = np.split(xe, [xe.shape[1]-ye.shape[1]], axis=1)
    
    return(xe, ye)
    
# Load parameters for SNN'straining
def load_cnf_snn(filename):
    par = np.genfromtxt(filename, delimiter=",")
    par_snn = []
    par_snn.append(par[0]) # Porcentaje de training
    par_snn.append(int(par[1])) # Numero maximo de iteraciones)
    par_snn.append(par[2]) # Taza de aprendizaje

    for i in range(3,par.shape[0]):
        par_snn.append(int(par[i])) # Nodos de las capas ocultas)
    
    return(par_snn)
   
# Beginning ...
def main():
    param       = load_cnf_snn("cnf_snn.csv")
    xe,ye       = load_data_trn("xData.csv","yData.csv")   
    W,Cost      = snn_train(xe,ye,param)             
    save_w(W,Cost)
       
if __name__ == '__main__':   
	 main()

