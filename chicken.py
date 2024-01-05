# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:34:38 2023
#code link: https://mbd.pub/o/liang/work
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class ChickenSwarmOptimizer:
    def __init__(self,num_chickens, num_iterations, lower_bound, upper_bound,search_space_dim,objective_function):
    #def __init__(self, objective_function, num_chickens=30, num_iterations=100, search_space_dim=20, lower_bound=-10, upper_bound=10):
        # 目标函数
        self.objective_function = objective_function
        # 鸡的总数
        self.num_chickens = num_chickens
        # 迭代次数
        self.num_iterations = num_iterations
        # 搜索空间维度
        self.search_space_dim = search_space_dim
        # 搜索空间的边界
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
       #code link: https://mbd.pub/o/liang/work
        # 将鸡分为公鸡、母鸡和小鸡
        self.num_roosters = int(num_chickens * 0.2)  # 公鸡数量
        self.num_hens = int(num_chickens * 0.7)     # 母鸡数量
        self.num_chicks = num_chickens - self.num_roosters - self.num_hens  # 小鸡数量
       #code link: https://mbd.pub/o/liang/work
        # 随机初始化鸡的位置
        self.positions = np.random.uniform(lower_bound, upper_bound, (num_chickens, search_space_dim))
        # 计算初始适应度
        self.fitness = np.array([self.objective_function(pos) for pos in self.positions])
       #code link: https://mbd.pub/o/liang/work


    def update_chicken_positions(self):
        # 根据适应度对鸡进行排序
        sorted_indices = np.argsort(self.fitness)
        roosters_indices = sorted_indices[:self.num_roosters]
        hens_indices = sorted_indices[self.num_roosters:self.num_roosters + self.num_hens]
        chicks_indices = sorted_indices[self.num_roosters + self.num_hens:]
       #code link: https://mbd.pub/o/liang/work
        # 更新鸡的位置
        for i in range(self.num_chickens):
            if i in roosters_indices:  # 公鸡行为
                self.positions[i] += np.random.uniform(-1, 1, self.search_space_dim)
            elif i in hens_indices:  # 母鸡行为
                rooster_index = np.random.choice(roosters_indices)
                self.positions[i] += np.random.uniform(-0.5, 0.5) * (self.positions[rooster_index] - self.positions[i])
            else:  # 小鸡行为
                mother_index = np.random.choice(hens_indices)
                self.positions[i] += np.random.uniform(-0.5, 0.5) * (self.positions[mother_index] - self.positions[i])
       #code link: https://mbd.pub/o/liang/work
            # 保持位置在边界内
            self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

    def optimize(self):
        curve=np.zeros(self.num_iterations)
        bestF=math.inf
        for iteration in range(self.num_iterations):
            self.update_chicken_positions()
            self.fitness = np.array([self.objective_function(pos) for pos in self.positions])

            # 找到最佳解
            best_fitness_index = np.argmin(self.fitness)
            best_position = self.positions[best_fitness_index]
            best_fitness = self.fitness[best_fitness_index]
            if best_fitness<bestF:
                bestF=best_fitness
                bestX=best_position
            curve[iteration]=bestF
       #code link: https://mbd.pub/o/liang/work    
            print(f"chicken 迭代 {iteration+1}: 最佳适应度 = {bestF}")
    
        return bestX, bestF,curve

def Chicken(pop,maxiteration,lb,ub,dim,fun):
    optimizer = ChickenSwarmOptimizer(pop,maxiteration,lb,ub,dim,fun)
    best_position, best_fitness,curve = optimizer.optimize()
    return best_position, best_fitness,curve
# 初始化并运行优化器

