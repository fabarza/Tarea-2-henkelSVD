import pandas     as pd
import numpy      as np


def hankel_features(par_prep, X):
  features = []
  M = X["samples"].shape[0] # Numero de muestras
  for i in range(0,M):
      x=X["samples"][i,:] # Toma la i-esima muestra
      sampSlices = np.split(x, x.shape[0]/par_prep[1]) # Separa la muestra en segmentos de tamaño par_prep[1] (L)
      k = par_prep[1]-par_prep[2]+1 # k = Largo del segmento - Cant de caracterisitcas + 1

      for slice in sampSlices:
        hankel = np.zeros((par_prep[2], k)) # Se crea la matriz de Hankel vacia

        for j in range(0, par_prep[2]): # Se llena la matriz de Hankel
          hankel[j,:] = slice[j:j+k]
        
        features.append(get_features(hankel)) # Se guardan las caracteristicas obtenidas de la matriz de hankel
        
        
  return() 

# Hankel's features
def get_features(mHankel):
  components = get_components(mHankel) # Se obtienen los componentes de la matriz de Hankel
  features = []

  for component in components:
    features.append(entropy_spectral(component))

  print(len(components))
  exit(0)
  pass


# Componentes de la matriz de Hankel
def get_components(mHankel):
  components = []
  u, s, vt = hankel_svd(mHankel) # Calcula la SVD de la matriz de Hankel

  print(u.shape, s.shape, vt.shape)
  for i in range(0, u.shape[0]):
    uReshape = np.reshape(u[:, i], (u.shape[0],1))
    vtReshape = np.reshape(vt[:, i], (1, vt.shape[0]))
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
  pass
  

# Binary Label
def binary_label():
  pass

# Data norm 
def data_norm():
  pass

# Save Data from  Hankel's features
def save_data_features(Dinp, Dout):
  pass

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
  data["labels"] = y

  return data

# Parameters for pre-proc.
def load_cnf_prep():
  par = np.genfromtxt("cnf_prep.csv", delimiter=",", dtype=int)
  par_prep = []
  par_prep.append(par[0])
  par_prep.append(par[1])
  par_prep.append(par[2])
  
  print(par_prep)
  return(par_prep)

# Beginning ...
def main():        
    par_prep    = load_cnf_prep()	
    Data        = load_data("Data_1.csv")	
    Dinput,Dout = hankel_features(par_prep, Data)
    Dinput      = data_norm(Dinput)
    save_data_features(Dinput,Dout)


if __name__ == '__main__':   
	 main()


