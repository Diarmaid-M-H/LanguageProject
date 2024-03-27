import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
sx = 0.5  # Prestige of Irish
c = 0.01  # Frequency of interaction
a = 1.23  # Volatility parameter

# Define the ODE system
def model(startProportions, t, sx, c, a):
    x, y = startProportions
    print("IRISH: ",x)
    print("ENGLISH: ",y)
    Pxy = c * (1 - sx) * (y ** a)
    #print("ProbXtoY: ", Pxy)
    Pyx = c * sx * (x ** a)
    #print("ProbYtoX: ", Pyx)
    dxdt = Pyx-Pxy  # change in x is the probability of increase - probability of decrease
    print("DX/DT: ", dxdt)
    dydt = -dxdt
    if x-dxdt <= 0:
        return [0,0]
    return [dxdt, dydt]

# Initial conditions
x0 = 0.01  # Initial proportion of bilingual Irish speakers
y0 = 1 - x0  # Initial proportion of monolingual English speakers

# Time points
t = np.linspace(0, 5, 50)  # Time span from 0 to 10 with 100 steps

# Solve the ODE system
solution = odeint(model, [x0, y0], t, args=(sx, c, a))

# Extract solution
x_solution, y_solution = solution[:, 0], solution[:, 1]

# Plot results

plt.figure(figsize=(10, 6))
plt.plot(t, x_solution, label='Bilingual Irish speakers')
plt.plot(t, y_solution, label='Monolingual English speakers')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('ODE Model: Bilingual vs. Monolingual')
plt.legend()
plt.grid(True)
plt.show()
