# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:50:04 2018

@author: murata
"""

from fractions import Fraction
from sympy import *

# 1)遷移行列 T に対する定常方程式を解く方法と、2)T を適当なベクトルに多数回当てて収束を見る方法の２つを比較する


""" 1)遷移行列 T に対する定常方程式を解く方法 """
print("1)遷移行列 T に対する定常方程式を解く方法")
#行列成分を分数として扱う
T = Matrix([[Fraction(1,20), Fraction(19,80), Fraction( 1,20), Fraction(1,5), Fraction( 1,20)],\
            [Fraction(3,10), Fraction( 1,20), Fraction(17,40), Fraction(1,5), Fraction( 1,20)],\
            [Fraction(3,10), Fraction(19,80), Fraction( 1,20), Fraction(1,5), Fraction(17,40)],\
            [Fraction(3,10), Fraction(19,80), Fraction( 1,20), Fraction(1,5), Fraction(17,40)],\
            [Fraction(1,20), Fraction(19,80), Fraction(17,40), Fraction(1,5), Fraction( 1,20)],\
            ])

#行列成分をfloatとして扱う
Tf = Matrix([[1/20, 19/80,  1/20, 1/5,  1/20],\
            [3/10,  1/20, 17/40, 1/5,  1/20],\
            [3/10, 19/80,  1/20, 1/5, 17/40],\
            [3/10, 19/80,  1/20, 1/5, 17/40],\
            [1/20, 19/80, 17/40, 1/5,  1/20],\
            ])

p1, p2, p3, p4, p5 = Symbol('p1'), Symbol('p2'), Symbol('p3'), Symbol('p4'), Symbol('p5')
P = Matrix([p1, p2, p3, p4, p5])

f_equilibrium = T*P - P
f_sum = p1+p2+p3+p4+p5 - 1

equilibrium_solution = solve([f_equilibrium, f_sum], P)

#p1, p2, p3, p4, p5 = 
p_equilibrium = [equilibrium_solution[p1], equilibrium_solution[p2], equilibrium_solution[p3], equilibrium_solution[p4], equilibrium_solution[p5]]
p_equilibrium_float = list(map(float, p_equilibrium))

print("平衡分布（分数)=",p_equilibrium)
print("平衡分布（小数)=",p_equilibrium_float)


""" 2)T を適当なベクトルに多数回当てて収束を見る方法 """
print("")
print("2)T を適当なベクトルに多数回当てて収束を見る方法")
pi_initial = Matrix([1,0,0,0,0])
print("pi(0) = ", pi_initial)
for n in [5, 10, 20, 50]:
    pi_converge = (Tf**n)*pi_initial
    print("T^%d pi(0)=" % n, pi_converge)
