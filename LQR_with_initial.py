import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from control.matlab import *

# System matrices
A = np.array([[0, 1, 0, 0],
              [0, -0.1818, 2.6727, 0],
              [0, 0, 0, 1],
              [0, -0.4545, 31.1818, 0]])

B = np.array([[0],
              [1.8182],
              [0],
              [4.5455]])

C = np.eye(4)
D = np.zeros((4,1))

# Cost matrices for LQR
Q = np.eye(4)
R = np.array([[1]]) * 20

# Compute the LQR controller
K, S, E = lqr(A, B, Q, R)

# System of equations
def system_eq(x, t):
    dxdt = (A - B @ K) @ x
    return np.squeeze(np.asarray(dxdt))

# Time vector
t = np.arange(0, 10, 0.01)

# Initial conditions
x0s = [[0, 0, 0, 0], [0.001, 0, 0, 0], [0, 0, 0.0001, 0]]

# Create figure and axis objects
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

for x0 in x0s:
    # Simulate the system
    x = odeint(system_eq, x0, t)

    # Plot the response
    ax1.plot(t, x[:, 0], label=f'Initial condition: {x0}')
    ax2.plot(t, x[:, 2], label=f'Initial condition: {x0}')

ax1.set_title('Cart Position with LQR Control for Different Initial Conditions')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cart Position (x)')
ax1.legend()
ax1.grid(True)

ax2.set_title('Pendulum Angle with LQR Control for Different Initial Conditions')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Pendulum Angle (phi)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(r'D:\github\LQR Controller\DoubleSidedArticle\simulations/high_r_rate.png')
plt.show()
