# Tarea 2: Matriz de Henkel y SVD

## Pasos
1) Crear [matiz de hankel](https://www.wikiwand.com/es/Matriz_de_Hankel) 
2) SVD de la matriz 
3) Extreamos los componentes


## Notas
### Paso 1:
### Paso 2:

### Paso 3: Matriz henkel SVD (de valores singulares)

* Podemos descomponer una matriz de henkel en series (1 serie por columna)
    * La suma de estas series debe ser igual a la matriz original (Descomposiscion de componentes intrinsecos)
    $$ X = C_1 + C_2 + C_3 $$
        * Con X = Serie de timepo original
        * C = Subseries de tiempo, componentes intrinsecos
    * Se utiliza para sacar el ruido de una señal

* Descomposicion diatica:
    * Creamos una matriz de henkel de 2 x N elementos
    * La primera serie que saquemos sera de baja fecuencia ($C_1$), la segunda sera de alta frecuencia ($C_2$)
        * El primer y ultimo elemento de la serie son iguales a la matriz X (Elemento $X_{11}$ y $X_{NN}$), pero los elementos de la mitad son la media de la matriz de la forma:
        [INSERTAR IMAGEN]
* s es una matriz diagonal 

* Entropia de shannon: Revisar esto

## Cuantificación de la informacion: Entropia de shannon
Se utiliza para medir la incertidumbre de una fuente de informacion
La transformada de fourier nos permite ver la informacion de la energia de los datos
* x: los datos
* el resto: modelo de euler
    * f: ciclos
    * t: tiempo (depende del largo de la data)
La transformada de fourier tambien se puede calcular de forma discreta
    * $(k*n)/N$ es 
    * A(k) es la amplitud
    * $f_0$ es la frecuendcia de auxicliacion
    * $f \in [0, (F_s/2)]$
La transformada de fourier permite identificar cuales son las frecuencias dominantes de una señal
Utilizamos la entropia espectral de shannon para identificar si la señal es deterministica

## Para la tarea:
* Van a haber 2 funciones distinas para la capa de salida y para las capas ocultas