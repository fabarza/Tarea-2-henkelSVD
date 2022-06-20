# Autores
# Felipe Abarzúa
# Stephanie Gómez
# Sergio Gil

import pandas as pd
import numpy as np
import my_utility as ut


def save_measure(cm,Fsc):
    np.savetxt("cm_snn.csv", cm, delimiter=",", fmt="%f")
    np.savetxt("Fsc_snn.csv", Fsc, delimiter=",", fmt="%f")

def load_w_snn(filename):
    W = np.load(filename, allow_pickle=True)
    return(W["W"])

def load_data_test(fileX, fileY):
    xv = np.genfromtxt(fileX, delimiter=",", dtype=np.float64)
    yv = np.genfromtxt(fileY, delimiter=",", dtype=np.float64)
    return(xv, yv)
    
# Beginning ...
def main():			
    xv,yv  = load_data_test("dTest_x.csv", "dTest_y.csv")
    W      = load_w_snn("w_snn.npz")
    zv     = ut.forward(xv,W)      		
    cm,Fsc = ut.metricas(yv,zv[-1])
    print(f"F-Score: {Fsc}")
    save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

