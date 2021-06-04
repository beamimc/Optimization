# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:57:52 2021

@author: es0055
"""

    
import json
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import pandas as pd
# import docplex.mp.solution as Solution
from pulp import *
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import pandas as pd
import docplex.mp.solution as Solution
from pulp import *
import time

if __name__ == '__main__':
    
    dist = pd.read_csv('Matriz_distancias_delegado_centroides_hospitales_referencia.csv', delimiter = ';', decimal = ',').iloc[:,1:].to_numpy()
    print('Matriz distancias: ', dist.shape)
    
    s = pd.read_csv('Ventas_Producto_y_Mdo_Hospital_Referencia.csv', delimiter = ';', decimal = ',')
    sp = s['VTAS_PRODUCTO'].to_numpy()
    sm = s['VTAS_MERCADO'].to_numpy()
    print('Vector ventas producto: ', sp.shape)
    print('Vector ventas mercado: ', sm.shape)
    
    
    hospitales = pd.read_csv(r'Matriz_distancias_delegado_centroides_hospitales_referencia.csv', delimiter = ';', decimal = ',').columns[1:]
    
    
    ## Crear modelo y meter parametros
    
    problem = LpProblem("TerritoryAlignment", LpMinimize)
    
    
    ## Variables de entrada: x(44,265) , y(44)
    
    x = LpVariable.dicts("x", ((i, j) for i in range(44) for j in range(265)), 0, 1, cat = LpInteger)
    y = LpVariable.dicts("y", (i for i in range(44) ), cat = LpContinuous)
    
    
    ## Se minimiza y   
     
    problem += (lpSum(y)/44) * 0.65 + (lpSum(x)/265) * 0.35
    
    
    ## Distancia <= 200km
    # En vez de poner x[i,j] = 0, guardamos las x de >200km y las
    # quitamos en las demás restricciones, como la de la suma de los
    # vendedores de un hospital = 1
    
    js = []
    
    for i in range(44):
      s = []
      for j in range(265):
          if dist[i, j] > 100:
              #problem += x[i,j] == 0
              s.append(j)
      js.append(s)
              
    
    ## Solo 1 vendedor por hospital.
    # No tenemos en cuenta las de >200km
    for j in range(265):
        if len([x[i,j] for i in range(44) if j not in js[i]]) > 1:
            problem += lpSum(x[i,j] for i in range(44) if j not in js[i]) <= 1
            
            
    ## Al menos 1 hospital por vendedor
    for i in range(44):
        if len([x[i,j] for j in range(265) if j not in js[i]]) > 1:
            problem += lpSum(x[i,j] for j in range(265) if j not in js[i]) >= 1
      
        
    ## 1 ecuacion de ventas
    # No tenemos en cuenta las de >200km
    for i in range(44):
      problem += 1 - lpSum([[( (sp[j] + sm[j]) / 2) * x[i,j]] for j in range(265) if j not in js[i]]) -y[i] <= 0
        
      
    ## 2 ecuacion de ventas
    # No tenemos en cuenta las de >200km
    for i in range(44):
      problem += 1 - lpSum([[( (sp[j] + sm[j]) / 2) * x[i,j]] for j in range(265) if j not in js[i]]) +y[i]  >= 0
      
      
    #Minimizar num hosp
    #problem += lpSum(x[i,j] for j in range(265)) / 265
    
    
    ## Imprimir y guardar restricciones del problema
    
    problem.constraints['_C2']
    with open('constraints.txt','w') as f:
      for c in problem.constraints.keys():
        print(problem.constraints[c])
        f.write(str(problem.constraints[c]) +  '\n')
    
    
    ## Resolver el problema
    
    start = time.time()
    problem.solve(PULP_CBC_CMD(timeLimit=240))
    end = time.time()
    
    print('Status: ', LpStatus[problem.status])
    
    
    ## Calcular ventas totales de la mejor solución obtenida
    
    import numpy as np
    l = []
    for i in range(44):
      for j in range(265):
          if x[i,j].varValue != None:    
              l.append(x[i,j].varValue)
          else:
              l.append(0)
    
    
    
    l = np.array(l)
    l = l.reshape((44, 265))
    
    np.sum(l)
    vent = []
    
    print('Mejor solución: ')
    
    for i in range(44):
        ventas = 0
        for j in range(265):
            ventas += (sm[j] * 0.5 + sp[j] * 0.5) * l[i, j]
    
        print(ventas)
        vent.append(ventas)
        
    
      
    obj = value(problem.objective)
    print('Fitness', obj)
    
    print(np.sum(vent))
    print(np.sum(l))
    
    df = pd.DataFrame(l, columns = hospitales)
    df.index = pd.read_csv(r'Matriz_distancias_delegado_centroides_hospitales_referencia.csv', delimiter = ';', decimal = ',').iloc[:,0]
    df.to_csv('sol_optima_065ventas_100dist.csv')
    
    import json
    
    problem.to_json("problem.json")
    
    print('Elapsed time:' , end - start)