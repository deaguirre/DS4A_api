from itertools import *
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

def calcular_Malla(data, model, variables, var1, var2, k:int):
    """ Crea una malla 2D para un modelo lineal
    Parameters:
        data -- el dataframe con el cual fue construido el modelo
        model -- modelo
        variables -- lista de variables incluidas en el modelo
        var1 -- variable 1 (eje X)
        var2 -- variable 2 (eje Y)
        k -- granularidad de la malla
    """
    # 1 - Seleccionar los datos a modelar en x y y
    x1 = data[var1]
    y1 = data[var2]
    # 2 - Generar una malla 2D con tamaño n
    X = np.linspace(x1.min(), x1.max(), k)
    Y = np.linspace(y1.min(), y1.max(), k)
    X, Y = np.meshgrid(X, Y)
    # 3 - Generar una array de diseño 3D con 
    A = [] # Inicializar
    A.append( np.ones((k,k), dtype=float) )
    A.append(X); A.append(Y) 
    # Primero se colocar una matriz K,K que corresponde al intercepto
    # Luego, adicionar las matrices correspondientes a las variables X y Y
    # 4 - Generar el vector de los parámetros
    beta = []
    beta.append( model.params['Intercept'] )
    beta.append( model.params[var1] )
    beta.append( model.params[var2] )
    # De manera similar, primero extraer parámetro de intercepto
    # Luego, adicionar los parámetros restantes

    for i, var in enumerate(variables):
        if var not in [var1, var2]:
            A.append( (np.ones((k,k), dtype=float) * data[var].mean()) )
            beta.append( model.params[var] )
    # Adicionar el resto de las variables como matrices que tienen todos 
    # los elementos iguales y corresponden a la media de la variable.
    # Luego, adicionar los parámetros restantes, excepto para X y Y.

    # 5. Hacer el cálculo de variable dependiente
    Z = np.sum(((np.array(A).T * np.array(beta))).T, axis=0)
    # 6. Retornar los componentes de la malla como dict
    return {'X':X, 'Y': Y, 'Z': Z}
    

def expand_malla(dict):
    """Crear una malla con vectores a partir de un diccionario
    Parameters:
        dict -- un diccionario compuesto de listas como valores
    """
    return pd.DataFrame([row for row in product(*dict.values())], columns=dict.keys())

##



def prediccionMallaModelo(data, model, variables, var1, var2, k=20, ofs_x=0, ofs_y=0, normal='default'):
    # 1 - Seleccionar los datos a modelar en x y y
    x1 = data[var1]
    y1 = data[var2]
    # 2 - Generar una malla 2D con tamaño n
    X = np.linspace(x1.min() * (1-ofs_x), x1.max() * (1+ofs_x), k)
    Y = np.linspace(y1.min() * (1-ofs_y), y1.max() * (1+ofs_y), k)
    X2, Y2 = np.meshgrid(X, Y)
    # 3 - Generar un DF como matriz de diseño 
    DF = pd.concat([
        pd.DataFrame(X2, index=range(k), columns=range(k)).reset_index().melt('index'),
        pd.DataFrame(Y2, index=range(k), columns=range(k)).reset_index().melt('index')], axis=1)\
    .iloc[:, [0,1,2,5]]
    
    DF.columns = ['pos_X','pos_Y', var1, var2] # Cambiar nombre de columnas
    
    DF_dict = dict()
    
    # 4 - Crear diccionario para adicionar variables que permanecen constantes
    for i, var in enumerate(variables):
        if var not in [var1, var2]:
            DF_dict[var] = repeat(data[var].mean(), DF.shape[0])
    
    # 5 - Concatenar variables fijas y dinámicas (de malla)
    DF1 = pd.concat([DF, pd.DataFrame(DF_dict)], axis=1).loc[: , chain([var1, var2], DF_dict.keys())]
    DF1 = DF1.reindex(variables, axis = 1)

    # 6 - Crear predicciones a partir de un modelo
    Z = np.array(model.predict(DF1)).reshape(k,k)

    if normal != 'default':
        return {'X':X, 'Y': Y, 'Z': Z}
        
    return {'X':X2, 'Y': Y2, 'Z': Z}