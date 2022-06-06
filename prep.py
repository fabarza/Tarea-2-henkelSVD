import pandas     as pd
import numpy      as np


def hankel_features(X):
  M = X.shape[0]  
  for i in range(0,M):
      x=X[i,:]
      # ...
      get_features()
  #...
  #...    
  return(Dinput,Doutput) 

# Hankel's features
def get_features():
    ...  
  return(...)


def hankel_svd():
    ...  
  return(...) 


# spectral entropy
def entropy_spectral():
  ...
  return 

# Binary Label
def binary_label():
  ...
  return

# Data norm 
def data_norm():
  ...
  return 

# Save Data from  Hankel's features
def save_data_features(Dinp, Dout):
    ...  
  return

# Load data from Data.csv
def load_data():
    ...
  return() 

# Parameters for pre-proc.
def load_cnf_prep():
  ...
  return 

# Beginning ...
def main():        
    par_prep    = load_cnf_prep()	
    Data        = load_data()	
    Dinput,Dout = hankel_features(...)
    Dinput      = data_norm(Dinput)
    save_data_features(Dinput,Dout)


if __name__ == '__main__':   
	 main()


