# -*- coding: utf-8 -*-
"""
博客地址：https://blog.csdn.net/weixin_46204734?type=blog
@author: IT猿手
"""

from CEC2013.cec2013 import *
from chicken import Chicken
import matplotlib.pyplot as plt
import numpy as np
import pso
import cso
import cso2


#主程序
function_name =19 #测试函数1-28
SearchAgents_no = 40#种群大小
Max_iter = 100#迭代次数
dim=50#维度 10/30/50/100
lb=-100*np.ones(dim)#下限
ub=100*np.ones(dim)#上限
cec_functions = cec2013(dim,function_name)
fobj=cec_functions.func#目标函数

# pso
curve_pso = pso.PS(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

# chicken
bestX, bestF, curve_chicken = Chicken(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

# cos
curve_cso = cso.CS(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

# cso2
curve_cso2 = cso2.CS2(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
#画收敛曲线图

plt.plot(curve_pso, color='red', label='pso')
plt.plot(curve_chicken, color='green', label='chicken')
plt.plot(curve_cso, color='blue', label='cso')
plt.plot(curve_cso2, color='purple', label='cso2')

plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Comparison on func' + str(function_name))
plt.legend()  # 添加图例
plt.grid(True)  # 添加网格线
plt.show()


