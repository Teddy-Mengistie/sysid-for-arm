from casadi import *
import numpy as np
import json

data = []
with open('characterization_data.json', 'r') as char_data:
	data = json.load(char_data)

N = len(data)
# now data = [{t, u1, u2, q1, q2, q1desired, q2desired}, ...]
# organize data
measured_x = np.zeros((2, N)) #[[q1, q2]]
measured_u = np.zeros((2, N)) #[[u1, u2]]

for i in range(N):
	datum = data[i]
	measured_x[:,i] = np.array([datum["q1"], datum["q2"]])
	measured_u[:,i] = np.array([datum["u1"], datum["u2"]])

opti = Opti()

p = opti.variable(5) # [kg11, kg12, kg22, ks1, ks2]
opti.subject_to(p[0] >= 0) # assert kg11 be positive
X = opti.variable(2, N)
opti.set_initial(X, measured_x)
U = opti.parameter(2, N)
opti.set_value(U, measured_u)


errorSum = 0
for i in range(N):
	q1 = X[0, i]
	q2 = X[1, i]
	u1_p = p[0] * cos(q1) + p[1] * cos(q1 + q2) + p[3] * sign(measured_u[0, i])
	u2_p = p[2] * cos(q1 + q2) + p[4] * sign(measured_u[1, i])
	error = sqrt((U[0, i] - u1_p)**2 + (U[1, i] - u2_p)**2)
	errorSum = errorSum + error

opti.minimize(errorSum)

opti.solver('ipopt')
opti.callback(lambda i : print(opti.debug.value(errorSum)))
sol = opti.solve()

