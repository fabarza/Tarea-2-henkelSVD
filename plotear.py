import pandas as pd
from matplotlib import pyplot as plt

# Plotear costo
df = pd.read_csv("cost_snn.csv")

plt.plot(df)
plt.show()