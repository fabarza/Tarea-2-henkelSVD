# SNN's Training :

import pandas     as pd
import numpy      as np
import my_utility as ut
    

#Save weights of the SNN
def save_w(W, cost):
    np.savez("w_snn.npz", W=W)
    np.savetxt("cost_snn.csv", cost, delimiter=",", fmt="%f")
    
#SNN's Training 
def snn_train(x,y,param):    
    W,V = ut.iniWs(x,y,param) # Le borre el S ya que creo que se usa para el ADAM
    Costo = []
    for iter in range(param[1]):    
        a       = ut.forward(x, W)
        gW,cost = ut.gradW(a, x, y, W)
        W,V  = ut.updW(gW,W, V, param[2])
        Costo.append(cost)
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
    pd.DataFrame(xe.T).to_csv("dTrain_x.csv", index=False, header=False)
    pd.DataFrame(ye.T).to_csv("dTrain_y.csv", index=False, header=False)
    pd.DataFrame(xv.T).to_csv("dTest_x.csv", index=False, header=False)
    pd.DataFrame(yv.T).to_csv("dTest_y.csv", index=False, header=False)

    return(xe.T, ye.T)
    
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

