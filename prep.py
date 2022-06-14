import pandas     as pd
import numpy      as np
import time


def hankel_features(par_prep, X):
  features = []
  labels = []
  M = X["samples"].shape[0] # Numero de muestras
  for i in range(0,M):
      x=X["samples"][i,:] # Toma la i-esima muestra
      sampSlices = np.split(x, x.shape[0]/par_prep[1]) # Separa la muestra en segmentos de tamaño par_prep[1] (L)
      
      k = par_prep[1]-par_prep[2]+1 # k = Largo del segmento - Cant de caracterisitcas + 1

      for r in range(len(sampSlices)):
        sliceFeatures = []
        hankel = np.zeros((par_prep[2], k)) # Se crea la matriz de Hankel vacia

        for j in range(0, par_prep[2]): # Se llena la matriz de Hankel
          hankel[j,:] = sampSlices[r][j:j+k]
        
        sliceFeatures.extend(get_features(hankel)) # Se guardan las caracteristicas obtenidas de la matriz de hankel
        sliceFeatures.append(entropy_spectral(sampSlices[r])) # Se guarda la entropia del segmento entero
        
        features.append(sliceFeatures)
        labels.append(X["labels"][i]) # Se guarda la etiqueta
        
  return(features, labels)

# Hankel's features
def get_features(mHankel):
  components = get_components(mHankel) # Se obtienen los componentes de la matriz de Hankel
  features = []

  for component in components:
    features.append(entropy_spectral(component)) # Entropia de los componentes

  u, s, vt = hankel_svd(np.array(components)) # Calcula la SVD de la matriz de Hankel

  features.extend(s.tolist())

  return (features)


# Componentes de la matriz de Hankel
def get_components(mHankel):
  components = []
  u, s, vt = hankel_svd(mHankel) # Calcula la SVD de la matriz de Hankel

  for i in range(0, u.shape[0]):
    uReshape = np.reshape(u[:, i], (u.shape[0],1))
    vtReshape = np.reshape(vt.T[:, i], (1, vt.T.shape[0]))
    hi = np.dot(uReshape, vtReshape) * s[i] # Calcula la i-esima caracteristica
    compVector = np.concatenate((hi[0, :], hi[1: , hi.shape[1]-1]), axis=None) # Primera fila + ultima columna

    components.append(compVector)

  return(components)


# Calcular SVD
def hankel_svd(hankel):
  """
  Retorna la SVD de una matriz de Hankel
  """
  return np.linalg.svd(hankel, full_matrices=False)


# spectral entropy
def entropy_spectral(component):
  # Entropia de shannon normalizada para cada componente
  sum = 0

  vectorFourier = np.abs(np.fft.fft(component))
  vectorFourier = vectorFourier[1:vectorFourier.shape[0]//2]
  
  totalE = np.sum(vectorFourier)

  for a_k in vectorFourier:
    sum += probabilityFunction(a_k, totalE) * np.log2(probabilityFunction(a_k, totalE))

  entropy = (-1/np.log2(vectorFourier.shape[0])) * sum
  
  return (entropy)
  
# Funcion de probabilidad para la entropia
def probabilityFunction(a_k ,totalEnergy):
  return(np.power(a_k, 2)/ totalEnergy)


# Binary Label
def binary_label(y):
  totalClases = np.unique(y).shape[0]
  y_bin = np.zeros((y.shape[0], totalClases), dtype=int)

  for i in range(0, y.shape[0]):
    y_bin[i, int(y[i])-1] = 1  

  return(y_bin)

# Data norm 
# REVISAR SI ESTA BIEN ESTA FUNCION
def data_norm(x):
  return (x - np.min(x)) / (np.max(x) - np.min(x))
  

# Save Data from  Hankel's features
def save_data_features(Dinp, Dout):
  pd.DataFrame(Dinp).to_csv("xData.csv", index=False, header=False)
  pd.DataFrame(Dout).to_csv("yData.csv", index=False, header=False)
  

# Load data from Data.csv
def load_data(filename):
  """
  Carga los datos de un archivo csv
  Returns:
    Data["sample"]: ndarray de muestras
    Data["label"]: ndarray de etiquetas
  """
  aux = np.genfromtxt(filename, delimiter=",", dtype=np.float64) # Carga de los datos
  y = aux[:,aux.shape[1]-1] # Toma la ultima columna
  samples = np.delete(aux, aux.shape[1]-1, axis=1)  # Toma todo menos la ultima columna

  data = {} # Diccionario con los datos
  data["samples"] = samples
  data["labels"] = binary_label(y)

  return data

# Parameters for pre-proc.
def load_cnf_prep():
  par = np.genfromtxt("cnf_prep.csv", delimiter=",", dtype=int)
  par_prep = []
  par_prep.append(par[0])
  par_prep.append(par[1])
  par_prep.append(par[2])

  return(par_prep)

# Beginning ...
def main():        
  # Measure time of function
    start_time = time.time()
    par_prep    = load_cnf_prep()	
    print(f"Elapsed time for load_cnf: {time.time() - start_time}")
    start_time = time.time()
    Data        = load_data("Data_1.csv")	
    print(f"Elapsed time for load_data: {time.time() - start_time}")
    start_time = time.time()
    Dinput,Dout = hankel_features(par_prep, Data)
    print(f"Elapsed time for hankel_features: {time.time() - start_time}")
    start_time = time.time()
    Dinput      = data_norm(Dinput)
    print(f"Elapsed time for data_norm: {time.time() - start_time}")
    save_data_features(Dinput,Dout)


if __name__ == '__main__':   
	 main()


