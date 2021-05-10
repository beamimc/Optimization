# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:24:38 2021

@author: es0055
"""
    
import pandas as pd
import numpy as np
import random
import statistics

#genetic algorithm 
class Chr():
    def __init__(self, 
                 df_dist, 
                 df_ventas, 
                 chrom = []):
        # self.genes = genes
        #inicialize solution 
        if len(chrom) == 0:
            chrom = [] # self.chrom
              # Hacemos cumplir restricciones:
              # 1- Distancia menor de 200km
              # 2- Sólo 1 vendedor por hospital
            max_dist = 200
            for j, h in enumerate(df_dist.columns[:]):
                possible = []
                l = df_dist[h].tolist()
                # print((l))
                for idx, i in enumerate(l):
                    if int(i) < max_dist:
                        possible.append(idx)
                if possible != []:
                    chrom.append(random.choice(possible))
                else:
                    chrom.append(-1)
        self.chrom = np.array(chrom)
        self.fit = 0
        self.posible_assigments = 0
        self.ventas = df_ventas
        self.dist = df_dist
        self.mean_assigments = 0
        self.mean_dist  = 0
        self.num_hosps = 0
        self.num_del = 0
        self.ventas_tot = 0
        self.mean_ventas = 0 
    def fitness(self):
        diff = 0
        # print('f')
        for i in  np.delete(np.unique(self.chrom),0):
            # print(i)
            diff += (abs(1-(self.ventas.VENTAS[self.chrom == i].sum())))**2
            # print(self.ventas.VENTAS[self.chrom == i].sum())
            #penalizar por num de hosp
            # diff +=0.01 * len(self.chrom[self.chrom == i])
            # print(diff)
        self.fit = diff
        return diff
    def metrics(self):
        assigned = self.chrom[self.chrom != -1]
        self.num_hosps = assigned.size
        self.num_del = np.unique(assigned).size
        mean_hosp=[]
        mean_ventas=[]
        self.ventas_tot = 0 
        for del_ in np.unique(assigned):
            mean_hosp.append(assigned[assigned==del_].size)
            mean_ventas.append(self.ventas.VENTAS[self.chrom == del_].sum())
        self.ventas_tot = np.array(mean_ventas).sum()
        self.mean_assigments = np.array(mean_hosp).mean()      
        self.mean_ventas = np.array(mean_ventas).mean()
        error_ventas = statistics.stdev(mean_ventas)
        return print(f'num of sales representatives = {self.num_del}\n\
total of assigned hospitals = {self.num_hosps}\n\
mean hospitals per sales representative = {self.mean_assigments}\n\
total sales = {self.ventas_tot}\n\
mean sales per sales representative = {self.mean_ventas} +- {error_ventas}')
            
        
    # def to_decimal(self):
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
        #inicialize pob 
        for i in range(self.num_chrom):
            self.pob.append(Chr(df_dist =  self.dist,
                                df_ventas =  self.ventas))

    def tournament_sel(self):
        # print('h')
        indiv = list(range(self.num_chrom)) * 2
        # print(indiv)
        selected = []
        while len(indiv) >=2:
            # print(indiv)
            i = indiv.pop(random.randint(0,len(indiv)-1))
            # print(i)
            j = indiv.pop(random.randint(0,len(indiv)-1))
            # print(j)
            # print(indiv)            
            fit_i = self.pob[i].fitness()
            fit_j= self.pob[j].fitness()
            if fit_i < fit_j:
                selected.append(self.pob[i])
            else:
                selected.append(self.pob[j])
        self.pob = selected
        return selected
    
    def crossover(self):
        # print(self.pob)
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
        self.pob = new_gen 
        return new_gen
 
    def mutation(self):
        for indiv in self.pob:
            prob = random.uniform(0, 1)
            # print(prob)
            # print(self.mut)
            # print('out')
            if prob <= self.mut:
                # print('in')
                index = random.randint(0,264)
                # print(index)
                max_dist = 200
                possible = [-1]                
                l = self.dist.iloc[:,index].tolist()
                for idx, i in enumerate(l):
                    if int(i) < max_dist:
                        possible.append(idx)
                #so theres no chance the prev value is chosen 
                if indiv.chrom[index] != -1:
                    possible.remove(indiv.chrom[index])
                # print(possible)
                # print(indiv.chrom[index])
                np.put(indiv.chrom, [index], [random.choice(possible)])
                # print(indiv.chrom[index])
        return self.pob
    def update_best(self):
        fitness_pob =[]
        for indiv in self.pob:
            
            fitness_pob.append(indiv.fitness())
        # print(fitness_pob)
        best = min(fitness_pob)
        # print(best)
        self.best_fitness = best
        self.best_sol = self.pob[fitness_pob.index(best)].chrom   

def generate_ventasdf(df_ventas, peso_prod = 0.5, peso_mdo = 0.5):
    new_df = pd.DataFrame(columns = ['HOSP_REF', 'VENTAS'])
    new_df['HOSP_REF'] = df_ventas['HOSP_REF']
    new_df['VENTAS'] = df_ventas.iloc[:,1]*peso_prod + \
                       df_ventas.iloc[:,2]*peso_mdo
    return new_df
def initial_solution(df_matrix_sol):    
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
    
    # c = GA(num_chrom=300, 
    #       df_dist = df_dist,
    #       df_ventas = df_ventas_tot,
    #       mut = 0.5, 
    #       cross = 0.6, 
    #       max_iters = 1000)
    # c.update_best()
    # iters = 0
    # while iters < 100:
    #     c.tournament_sel()
    #     c.crossover()
    #     c.mutation()
    #     c.update_best()
    #     print(c.best_fitness)
    #     print(iters)
    #     iters +=1
    