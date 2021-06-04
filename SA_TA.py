# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:09:46 2021

@author: es0055
"""
#simulated anealling 
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time

class SA():
    def __init__(self, df_dist, df_ventas, max_dist, peso_ventas):
        self._actual_sol =[]
        self.actual_fit = 0
        self.best_sol = []
        self.best_fit = 0
        self._new_sol = []
        self.new_fit = 0
        self.max_dist = max_dist
        self.peso_ventas = peso_ventas
        self.df_dist = df_dist
        self.df_ventas = df_ventas
    @property 
    def actual_sol(self):
        return self._actual_sol
    @actual_sol.setter
    def actual_sol(self, sol):
        self._actual_sol = sol
        self.actual_fit = self.fitness_function(sol)
    @property 
    def new_sol(self):
        return self._new_sol
    @new_sol.setter
    def new_sol(self, sol):
        self._new_sol = sol
        self.new_fit = self.fitness_function(sol)
    def generate_initial_solution(self):
        # Inicializamos solución
        sol = np.zeros((44, 265))
        count = 0
        for j, h in enumerate(df_dist.columns[1:]):
          possible = []
          l = df_dist[h].tolist()
          for idx, i in enumerate(l):
            if int(i) < self.max_dist:
              possible.append(idx)
          if possible != []:
            sol[random.choice(possible)][j] = 1
            count += 1
        self.actual_sol = sol.copy()
        return self.actual_sol
    def new_solution(self, num_changes):
        new_sol = self.actual_sol.copy()
        # Poner a 0 alguna columna random
        for i in range(num_changes):
            new_sol[:, random.randint(0,264)] = 0
        # Poner un 1 en alguna columna random
        for i in range(num_changes):
            fila = random.randint(0,43)
            col = random.randint(0,264)
          # Ver si esta a menos de 200 km
            if df_dist.iloc[fila, col+1] <= self.max_dist:
                if np.sum(new_sol[:, col]) == 0:
                    new_sol[fila, col] = 1
                else:
                    new_sol[:, col] = 0
                    new_sol[fila, col] = 1
        self.new_sol = new_sol
        return new_sol
    def ventas_solucion(self, sol):
        # [peso producto, peso mercado]
        pesos = [0.5, 0.5]
        ventas_prod = self.df_ventas['VTAS_PRODUCTO'].tolist()
        ventas_mdo = self.df_ventas['VTAS_MERCADO'].tolist()
        ventas_prod = [i * pesos[0] for i in ventas_prod]
        ventas_mdo = [i * pesos[1] for i in ventas_mdo]
        ventas = [x + y for x, y in zip(ventas_prod, ventas_mdo)]
        v = []
        for row in sol:
            ventas_delegado = [a * b for a, b in zip(ventas, row)]
            v.append(sum(ventas_delegado))
        return v
    def fitness_function(self, sol):
        pventas = self.peso_ventas
        phosp = 1-pventas
        ventas = self.ventas_solucion(sol)
        diff = 0
        ## Objetivo 1: minimizar numero de hospitales
        diff_1 = np.sum(sol) / 265 # se normaliza para que sea de 0 a 1
        ## Objetivo 2: minimizar dif a 1 las ventas de cada vendedor
        diff_2 = 0
        for v in ventas:
          diff_2 += abs(1-v)
        diff_2 = diff_2 / 44 # se normaliza para que sea de 0 a 1
        #funcion uniobjetivo con pesos para cada objetivo
        diff = diff_2*pventas + diff_1*phosp
        return diff
def main(df_dist, df_ventas, max_dist, peso_ventas, num_changes, t_ini, alpha, L, epsilon, k):
        global solutions
        global best_solutions
        global prob
        global sa 
        prob = []
        prob_worse = []
        solutions = []
        best_solutions = []
        #inicializo SA
        sa = SA(df_dist=df_dist,
                df_ventas=df_ventas,
                max_dist = max_dist, 
                peso_ventas = peso_ventas )
        sa.generate_initial_solution()
        sa.best_fit = sa.actual_fit
        sa.best_sol = sa.actual_sol.copy()
        solutions.append(sa.actual_fit)
        t = t_ini # Current temp
        ls=[]
        parada = False
        i=1 #para contar los ciclos
        acepted = [] #para el criterio de parada 3
        while not parada: 
            innit= time.time()
            print(np.array(prob_worse).mean())
            print('Cycle: ' , str(i), ' - Temperature = ', str(t))
            for j in range(L):
                sa.new_solution(num_changes=num_changes)
                if (sa.new_fit <  sa.actual_fit):
                    if sa.new_fit < sa.best_fit:
                        sa.best_fit = sa.new_fit
                        sa.best_sol = sa.new_sol.copy()
                    sa.actual_sol = sa.new_sol.copy()
                    acepted.append(1) #para el criterio de parada 3
                    prob.append(1)
                else:
                    p = math.exp( (- sa.new_fit + sa.actual_fit)/ t )
                    prob.append(p)
                    prob_worse.append(p)
                    u = random.uniform(0,1)
                    if u < p:
                        sa.actual_sol = sa.new_sol.copy()
                        acepted.append(1) #para el criterio de parada 3
                    else:
                        acepted.append(0) #para el criterio de parada 3
                solutions.append(sa.actual_fit)
                best_solutions.append(sa.best_fit)
                
            ls.append(sa.best_fit)
            if len(ls) > 2:
                print((1-(ls[-1]/ls[-2]))*100, '% de mejora')
               
                # ##criterio de parada 1
                # if sa.actual_fit == 0:
                #     parada = True
                #     break
                # criterio 3:
                # eps = epsilon*L
                # kl = k*L
                # if len(acepted)>kl and (sum(acepted[-kl:])<eps):
                #     parada = True
                #     break
                # ##criterio 2
                # kl = k*L
                # if len(best_solutions)>kl:
                #     if (1-(best_solutions[-1]/best_solutions[-kl])) < epsilon:
                #         parada = True
                #         break
                ##criterio dani
            ss = []
            if len(ls) >= k:
                for ii in range(k):
                    ## Si empeora, añado un 0 a ss
                    if (1 - (ls[-k + ii + 1] / ls[-k + ii]) < epsilon):
                        ss.append(0)
                if len(ss) == k:
                    parada = True
                    break
            print(len(ss), ' soluciones han empeorado al menos ', epsilon*100, '%')
                    
            t = alpha * t #disminuyo temp
            i +=1 #incremento el nº de ciclo por el q voy
            
            end= time.time()
            print('time cicle', end-innit)
        ##solucion optenida
        print(f"""\nParámetros de ejecución:\n max_dist: {max_dist}\
              \n peso ventas: {peso_ventas}\n num_changes: {num_changes}\
                  \n T inicial: {t_ini} \
               \n alpha: {alpha}\n L: {L} \n epsilon: {epsilon} \n k: {k}\n""")
        print(f'Best fitness: {sa.actual_fit}')
        print(f'Total hospitales: {np.sum(sa.actual_sol)}')
        print(f'Total ventas: {np.sum(sa.ventas_solucion(sa.actual_sol))}')
        print(f'Media de hospitales por vendedor: {np.sum(sa.actual_sol)/44}')
        print(f'Media de ventas por vendedor: {np.array(sa.ventas_solucion(sa.best_sol)).mean()}')
        print(f'Media de distancia a cada hospital: {np.sum((df_dist.iloc[:,1:] * sa.actual_sol).values)/np.sum(sa.actual_sol)}')
        print()
        
if __name__ == '__main__':
    df_dist = pd.read_csv('Matriz_distancias_delegado_centroides_hospitales_referencia.csv', 
                          delimiter = ';', 
                          decimal = ',')
    df_ventas = pd.read_csv('Ventas_Producto_y_Mdo_Hospital_Referencia.csv',
                            delimiter = ';', 
                            decimal = ',')
    

    
    ###########################################
    ##### Ejecucion recocido simulado ############
    start = time.time()
    optimized_result = main(df_dist = df_dist,
                            df_ventas= df_ventas,
                            max_dist=200,
                            peso_ventas =0.65,
                            num_changes=4,
                            t_ini= 0.03, 
                            alpha=0.98,
                            L=200,
                            epsilon=0.0005,
                            k=500)
    end = time.time()
    print(f'Tiempo de ejecución: {end-start}')
    
    
    ################################################
    ######### Plots #############
    plt.style.use('seaborn-pastel')
    fig = plt.figure(figsize=(12,6))
    # ax1 = fig.add_subplot(111)
    plt.scatter(x = list(range(len(solutions))),
                y = solutions, 
                marker = '.', 
                color = '#7070E7', 
                s= 2,
                label = 'Fitness de la solucion escogida en cada iteración')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')    
    plt.title('Fitness de la solucion escogida en cada iteración')
    plt.show()
    
    plt.figure(figsize=(12,6))
    # ax1 = fig.add_subplot(111)
    plt.scatter(x = list(range(len(best_solutions))),
                y = best_solutions, 
                marker = '.', 
                color = '#E45858', 
                s= 2,
                label ='Fitness de la mejor solución hasta el momento' )
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')    
    plt.title('Fitness de la mejor solución hasta el momento')
    # plt.legend(markerscale=8, loc = 'upper right',  prop={'size': 10})
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.scatter(x = list(range(len(prob))),
                y = prob, 
                marker = '.', 
                color = '#F9A526', 
                s= 2)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')    
    plt.title('Probabilidad de aceptación')
    plt.show()

    v_del = sa.ventas_solucion(sa.best_sol)
    plt.figure(figsize=(12,12))
    plt.barh(list(range(len(v_del))),
                v_del)
    plt.axvline(x = 1, color = '#E45858', linestyle= 'dashed') 
 
    plt.ylabel('Delegado')
    plt.xlabel('% Ventas')    
    plt.title('%ventas de cada delegado')
    plt.yticks(np.arange(0, 44, 1))
    plt.show()
    
    # print(f'Best fitness: {sa.actual_fit}')
    # print(f'Total hospitales: {np.sum(sa.actual_sol)}')
    # print(f'Total ventas: {np.sum(sa.ventas_solucion(sa.actual_sol))}')
    # print(f'Media de hospitales por vendedor: {np.sum(sa.actual_sol) / 44}')
    # print(f'Media de ventas por vendedor: {np.array(sa.ventas_solucion(sa.best_sol)).mean()}')
    # print(f'Media de distancia a cada hospital: {np.sum((df_dist.iloc[:,1:] * sa.actual_sol).values)/np.sum(sa.actual_sol)}')
    # print()
