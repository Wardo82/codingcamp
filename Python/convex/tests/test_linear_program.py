""" Simple example of lienar program optimization for the Farmer's problem 

"""
from pulp import *

# Problem formulation
model = LpProblem(sense=LpMaximize)

x_p = LpVariable(name="potatoes", lowBound=0)
x_c = LpVariable(name="carrots", lowBound=0)

model += x_p       <= 3000 # potatoes
model +=       x_c <= 4000 # carrots
model += x_p + x_c <= 5000 # fertilizer

model += x_p * 1.2 + x_c * 1.7

# Solve
status = model.solve(PULP_CBC_CMD(msg=False))
print("Potatoes:", x_p.value())
print("Carrots:", x_c.value())
print("Profit:", model.objective.value()) 