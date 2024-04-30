# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:37:16 2022

@author: Akash
"""

import pandas as pd
import pulp as pl

file_name='model.xlsx'
df = pd.read_excel(file_name, "Sheet1",
                   index_col=0).fillna(0)  


product = df.loc[df.index[-1], df.columns[0:-1]].to_dict()


constraint_matrix = pd.DataFrame(
    df, index=df.index[0:-1], columns=df.columns[0:-1]).to_dict('index')

rhs_coefficients = df.loc[df.index[0:-1], df.columns[-1]].to_dict()


model1 = pl.LpProblem("Sheet1", pl.LpMinimize)


variables = pl.LpVariable.dicts('amount', product, lowBound=0)

model1 += pl.lpSum([product[i]*variables[i] for i in product])

for c in rhs_coefficients: 
    model1 += (pl.lpSum([constraint_matrix[c][u]*variables[u] for u in product]) 
        <= rhs_coefficients[c], c) 

model1.solve() 

print("Status:", pl.LpStatus[model1.status])

print("Total Cost = ", pl.value(model1.objective))
if (pl.LpStatus[model1.status] == 'Optimal'):

    for v in model1.variables():
        print(v.name, "=", v.varValue)

print('\n ADVANCED: Info about the constraints for the solution found')
for name, constraint in model1.constraints.items():
    print(f"{name}: {constraint.value():.2f}")
model1.writeLP("Linear_Problem.lp")