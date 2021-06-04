# -*- coding: utf-8 -*-
"""
Created on Sun May  9 22:44:45 2021

@author: es0055
"""

import pandas as pd
import numpy as np

sol = pd.read_csv('matriz_solucion_inicial_mundi_TA.csv',
            delimiter = ';',
            decimal = ',')
sol.drop(['DELEGADO'], axis = 1, inplace =True)


solution = []

for i in range(0,265):
    aux = np.array(sol.iloc[:,i])
    del_index = np.where(aux == 1)
    if len(del_index[0]) == 1:
        print(del_index)
        solution.append(del_index[0][0])
    else:
        solution.append(-1)
    