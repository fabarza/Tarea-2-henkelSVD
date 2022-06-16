# SNN's Training :

import pandas     as pd
import numpy      as np
import my_utility as ut


#Inicializacion de pesos 
def iniWs(x,y,param_snn):
    W = []
    V = [np.zeros(x.shape[1])]
    prev = x.shape[1]
    # Inicializacion de pesos de la capa de entrada
    for i in range(3, len(param_snn)):
        next = param_snn[i]
        W.append(ut.iniW(prev, next))  # Guardamos los pesos por capa
        prev = next
        # Asumo que cuando pide guardar el shape se refiere a la cantidad de filas
        V.append(np.zeros(W[i-3].shape[0])) # Velocidad (DUDOSO, preguntar al profe)        
    W.append(ut.iniW(prev, y.shape[1])) # Pesos de la capa de salida

    return (W, V)
    

#Save weights of the SNN
def save_w():
    pass 
    
#SNN's Training 
def snn_train(x,y,param):    
    W,V = iniWs(x,y,param) # Le borre el S ya que creo que se usa para el ADAM
    Costo = []
    for iter in range(param[1]):    
        a       = ut.forward(x, W)
        gW,cost = ut.gradW(a, x, W, V)
        W,V  = ut.updW(...)
        Costo.append(Cost)
    return(W,Costo)

# Load data from xData.csv, yData,csv
def load_data_trn(xData,yData, trainPer):
    rng = np.random.default_rng() # Generador de numeros aleatorios para shuffle
    # Lectura de datos
    xe = np.genfromtxt(xData, delimiter=",", dtype=np.float64)
    ye = np.genfromtxt(yData, delimiter=",", dtype=np.float64)

    xe = np.append(xe, ye, axis=1) # Union de las caracteristicas con las etiquetas
    rng.shuffle(xe) # Desordenar las filas

    # Separacion de datos de entrenamiento y prueba
    xe_trn = xe[:int(xe.shape[0]*trainPer),:]
    xe_tst = xe[int(xe.shape[0]*trainPer):,:]

    # Separo los datos de las etiquetas
    xe, ye = np.split(xe_trn, [xe_trn.shape[1]-ye.shape[1]], axis=1)
    xv, yv = np.split(xe_tst, [xe_tst.shape[1]-ye.shape[1]], axis=1)

    # Guardamos los archivos de entrenamiento y prueba
    pd.DataFrame(xe).to_csv("dTrain_x.csv", index=False, header=False)
    pd.DataFrame(ye).to_csv("dTrain_y.csv", index=False, header=False)
    pd.DataFrame(xv).to_csv("dTest_x.csv", index=False, header=False)
    pd.DataFrame(yv).to_csv("dTest_y.csv", index=False, header=False)
    
    return(xe, ye)
    
# Load parameters for SNN'straining
def load_cnf_snn(filename):
    par = np.genfromtxt(filename, delimiter=",")
    par_snn = []
    par_snn.append(par[0]) # Porcentaje de training (Tamaño del conjunto de entrenamiento)
    par_snn.append(int(par[1])) # Numero maximo de iteraciones
    par_snn.append(par[2]) # Taza de aprendizaje

    for i in range(3,par.shape[0]):
        par_snn.append(int(par[i])) # Nodos de las capas ocultas
    
    return(par_snn)
   
# Beginning ...
def main():
    param       = load_cnf_snn("cnf_snn.csv")
    xe,ye       = load_data_trn("xData.csv","yData.csv", param[0])
    W,Cost      = snn_train(xe,ye,param)             
    save_w(W,Cost)
       
if __name__ == '__main__':   
	 main()

