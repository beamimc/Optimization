# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:24:38 2021

@author: es0055
"""
    
import pandas as pd
import numpy as np
import random
import statistics
import matplotlib.pyplot as plt

#genetic algorithm 
class Chr():
    def __init__(self, 
                 df_dist, 
                 df_ventas, 
                 max_dist = 200,
                 chrom = []):
        self.ventas = df_ventas
        self.dist = df_dist
        self.possible_assignments = []
        self.dist_ter = []
        self.hosp_ter = []
        self.ventas_ter = []
        for j, h in enumerate(self.dist.columns[:]):
            possible = [-1]
            l = df_dist[h].tolist()
            for idx, i in enumerate(l):
                if int(i) < max_dist:
                    possible.append(idx)
            self.possible_assignments.append(possible)
            
        #inicialize solution 
        if len(chrom) == 0:
            chrom = [] # self.chrom
            for i in range(0, self.dist.columns.size):
                chrom.append(random.choice(self.possible_assignments[i]))
        self._chrom = np.array(chrom)
        self.fit = self.fitness()
    @property
    def chrom(self):
        return self._chrom
    @chrom.setter
    def chrom(self, chrom):
        self._chrom = chrom
        self.fit = self.fitness()
        # self.metrics()
    def fitness(self):
            diff1 = 0
            # print('f')
            for i in  np.delete(np.unique(self.chrom),0):
                # print(i)
                diff1 += abs(1-(self.ventas.VENTAS[self.chrom == i].sum()))
                diff1 = diff1/44
                # print(self.ventas.VENTAS[self.chrom == i].sum())
                #penalizar por num de hosp
                # diff +=0.01 * len(self.chrom[self.chrom == i])
                # print(diff)
            self.fit = diff1
            return diff1
    def metrics(self):
        assigned = self.chrom[self.chrom != -1]
        self.hosp_ter =[]
        self.ventas_ter=[]
        for i in range(0,44):
            if i in np.unique(assigned):
                mask = self.chrom==i
                self.hosp_ter.append(assigned[assigned==i].size)
                self.ventas_ter.append(self.ventas.VENTAS[mask].sum())
                self.dist_ter.append(((self.dist.iloc[i, mask]).sum())/len(self.dist.iloc[i, mask])) 
            else:
                self.hosp_ter.append(0)
                self.ventas_ter.append(0)
                self.dist_ter.append(0)
        return 
    #     self.dec_chrom = []
    #     print(self.chrom.shape[1])
    #     for i in range(self.chrom.shape[0]):
        
    #         # Convert a binary columns to a decimal int.
    #         column_str = ''.join([str(round(elem)) \
    #                               for elem in self.chrom[i,:]])
    #         self.dec_chrom.append(int(column_str, 2)) 
    # def to_matrix(self):
    #     self.chrom = []
    #     for i in range(len(self.dec_chrom)):
    #         row = bin(self.dec_chrom[i]).replace("0b", "")
    #         row = row.rjust(265,'0')            
    #         self.chrom.append(list(map(int, row)))
    #     self.chrom = np.array(self.chrom)
    #     print(self.chrom.shape)
class GA():
    def __init__(self, 
                  num_chrom, 
                  df_dist,
                  df_ventas,
                  mut = 0.2, 
                  cross = 0.4, 
                  max_iters = 1000):
        self.ventas = df_ventas
        self.dist = df_dist
        self.num_chrom = num_chrom
        self.mut = mut
        self.cross = cross
        self.max_iters = max_iters
        self.best_sol = []
        self.best_fitness = 0
        self.pob = []
        self.percent_with_best_fit = 0
        #inicialize pob 
        for i in range(self.num_chrom):
            self.pob.append(Chr(df_dist =  self.dist,
                                df_ventas =  self.ventas))

    def tournament_sel(self):
        indiv = list(range(self.num_chrom)) * 2
        selected = []
        while len(indiv) >=2:
            i = indiv.pop(random.randint(0,len(indiv)-1))
            j = indiv.pop(random.randint(0,len(indiv)-1))
            fit_i = self.pob[i].fitness()
            fit_j= self.pob[j].fitness()
            if fit_i < fit_j:
                selected.append(self.pob[i])
            else:
                selected.append(self.pob[j])
        self.pob = selected
        return selected
    
    def crossover(self):
        num_cross = self.cross*self.num_chrom
        new_gen = []
        if num_cross%2 != 0 :
            num_cross -= 1
        while num_cross > 0:
            parent1 = self.pob.pop(random.randint(0,len(self.pob)-1))
            parent2 = self.pob.pop(random.randint(0,len(self.pob)-1))
            child1 = np.append(parent1.chrom[:100], parent2.chrom[100:])
            child2 = np.append(parent2.chrom[:100], parent1.chrom[100:])
            new_gen.append(Chr(df_dist = self.dist,
                                df_ventas = self.ventas,
                                chrom = child1))
            new_gen.append(Chr(df_dist = self.dist,
                                df_ventas = self.ventas,
                                chrom = child2))
            num_cross -= 2
        new_gen.extend(self.pob)
        self.pob = new_gen.copy()
        return new_gen
    def mutation(self):
        for indiv in self.pob:
            prob = random.uniform(0, 1)
            if prob <= self.mut:
                index = random.randint(0,264)
                max_dist = 200
                possible = [-1]                
                l = self.dist.iloc[:,index].tolist()
                for idx, i in enumerate(l):
                    if int(i) < max_dist:
                        possible.append(idx)
                #so theres no chance the prev value is chosen 
                if indiv.chrom[index] != -1:
                    possible.remove(indiv.chrom[index])
                np.put(indiv.chrom, [index], [random.choice(possible)])
        return self.pob
    def update_best(self):
        fitness_pob = []
        for indiv in self.pob:
            fitness_pob.append(indiv.fitness())
        best = min(fitness_pob)
        self.best_fitness = best
        self.best_sol = self.pob[fitness_pob.index(best)]
        fitness_pob = np.array(fitness_pob)
        self.percent_with_best_fit = len(fitness_pob[fitness_pob == best])/ self.num_chrom

def generate_ventasdf(df_ventas, peso_prod = 0.5, peso_mdo = 0.5):
    new_df = pd.DataFrame(columns = ['HOSP_REF', 'VENTAS'])
    new_df['HOSP_REF'] = df_ventas['HOSP_REF']
    new_df['VENTAS'] = df_ventas.iloc[:,1]*peso_prod + \
                       df_ventas.iloc[:,2]*peso_mdo
    return new_df
def initial_solution():    
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
    return(solution)

if __name__ == '__main__':    
    df_dist = pd.read_csv('Matriz_distancias_delegado_centroides_hospitales_referencia.csv', 
                          delimiter = ';', 
                          decimal = ',')
    df_ventas = pd.read_csv('Ventas_Producto_y_Mdo_Hospital_Referencia.csv', 
                            delimiter = ';', 
                            decimal = ',')
    delegados = df_dist['DELEGADO'].tolist()
    df_dist.drop(['DELEGADO'], axis=1, inplace = True)
    
    df_ventas_tot = generate_ventasdf(df_ventas)
    
    parada = False
    solutions = []
    percent =[]
    k = 10
    eps = 0.01
    ga = GA(num_chrom=300, 
          df_dist = df_dist,
          df_ventas = df_ventas_tot,
          mut = 0.5, 
          cross = 0.6, 
          max_iters = 100)
    ga.update_best()
    iters = 0
    while not parada:
        ga.tournament_sel()
        ga.crossover()
        ga.mutation()
        ga.update_best()
        solutions.append(ga.best_fitness)
        percent.append(ga.percent_with_best_fit)
        print(f'iter: {iters}, best fitness: {ga.best_fitness}')
        iters +=1
        
        # ##criterio parada 1:
        # if iters == ga.max_iters:
        #     parada = True
        #criterio 2:
        if len(solutions)>k and (1-(solutions[-1]/solutions[-k])) < eps:
            parada = True

    
#     assigned = ga.best_sol.chrom[ga.best_sol.chrom != -1]        
#     #assignment metrics 
#     total_hosp_assigned = assigned.size
#     total_del_used = np.unique(assigned).size
#     mean_assigments = np.array(ga.best_sol.hosp_ter).mean() 
#     #sales metrics
#     total_ventas = np.array(ga.best_sol.ventas_ter).sum()
#     mean_ventas = np.array(ga.best_sol.ventas_ter).mean()
#     error_ventas = statistics.stdev(ga.best_sol.ventas_ter)
#     print(f'num of sales representatives = {total_del_used}\n\
# total of assigned hospitals = {total_hosp_assigned}\n\
# mean hospitals per sales representative = {mean_assigments}\n\
# total sales = {total_ventas}\n\
# mean sales per sales representative = {mean_ventas} +- {error_ventas}')        
    
    plt.style.use('seaborn-pastel')
    plt.figure(figsize=(12,6))
    plt.scatter(x = list(range(len(solutions))),
                y = solutions, 
                marker = '.', 
                color = '#7070E7', 
                s= 1)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')    
    plt.title('Fitness de la mejor solucion de cada iteración')
    plt.show()
    
    plt.style.use('seaborn-pastel')
    plt.figure(figsize=(12,6))
    plt.scatter(x = list(range(len(percent))),
                y = percent, 
                marker = '.', 
                color = '#E45858', 
                s= 1)
    plt.xlabel('Iterations')
    plt.ylabel('% of solutions')    
    plt.title('Porcentaje de soluciones con el mejor fitness de cada iteración')
    plt.show()
