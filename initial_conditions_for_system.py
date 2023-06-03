from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# system matrices
A = np.array([[0, 1, 0, 0],
              [0, -0.1818, 2.6727, 0],
              [0, 0, 0, 1],
              [0, -0.4545, 31.1818, 0]])

B = np.array([0, 1.8182, 0, 4.5455])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

D = np.array([0, 0])

# system dynamics
def system_dynamics(y, t):
    return A.dot(y) + B*u

# initial conditions
initial_conditions = [[0, 0, 0, 0], [0.001, 0, 0, 0], [0, 0, 0.0001, 0]]

# simulation time
time = np.linspace(0, 5, 5000)  # simulate for 5 seconds
u = 0  # assuming control input u = 0

# for nicer plots
sns.set(style="whitegrid")
plt.figure(figsize=(12,8))

# loop over initial conditions
for i, init in enumerate(initial_conditions):
    # solve ODE
    y = odeint(system_dynamics, init, time)

    # plot results
    plt.subplot(2, 1, 1)
    plt.plot(time, y[:, 0], label=f'Initial conditions {init}')
    plt.title('Cart Position vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, y[:, 2], label=f'Initial conditions {init}')
    plt.title('Pendulum Angle vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()

plt.tight_layout()


# save to file
plt.savefig(r'D:\github\LQR Controller\DoubleSidedArticle\simulations/system_simulation.png')