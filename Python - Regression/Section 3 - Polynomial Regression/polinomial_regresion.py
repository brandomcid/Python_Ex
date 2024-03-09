# -*- coding: utf-8 -*-

#Polinomic regression

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X, y)


#Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizacion de resultados modelo lineal
plt.scatter(X, y, color = "red")
plt.plot(X,regression.predict(X), color = "blue")
plt.title("Modelo de regresion lineal")
plt.title("Posicion de empleado")
plt.ylabel("Sueldo en $")
plt.show()

#Visualizacion de resultados modelos polinomico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de regresion polinomico")
plt.title("Posicion de empleado")
plt.ylabel("Sueldo en $")
plt.show()

#Prediccion de modelos
prediction = regression.predict([[6.5]])
prediction2 = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))








