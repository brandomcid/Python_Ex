# -*- coding: utf-8 -*-

#Paso uno es importar librerias
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

"""#Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)"""

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#Ajustar la regresion con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)

#Prediccion de modelos
y_pred = regression.predict(sc_X.transform([[6.5]])).reshape(-1, 1)
y_pred_inverse = sc_y.inverse_transform(y_pred)

#Plot
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de regresion SVR")
plt.xlabel("Posicion de empleado")
plt.ylabel("Sueldo en $")
plt.show()



