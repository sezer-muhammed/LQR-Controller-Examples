import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are as care
from scipy.integrate import odeint
from lqr_controller import LQRController

# Define the system and cost matrices
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

# Define the cost matrices
Q_normal = np.eye(4)
R_normal = np.eye(1)
Q_high = np.zeros((4, 4))  # 10 times higher
Q_high[0][0] = 100
R_high = 10 * np.eye(1)  # 10 times higher

# Create instances of the LQRController
lqr_normal = LQRController(A, B, Q_normal, R_normal)
lqr_highQ = LQRController(A, B, Q_high, R_normal)
lqr_highR = LQRController(A, B, Q_normal, R_high)

# LQR controllers in a list for easy iteration
lqrs = [lqr_normal, lqr_highQ, lqr_highR]
labels = ['Normal Q, R', 'High Positonal Q', 'High R']

# System dynamics
def system_dynamics(x, t, u):
    dxdt = A @ x + B @ u
    return dxdt

# Time vector
T = np.linspace(0, 30, 10000)  # simulation for 12 seconds

# Setpoints and initial condition
setpoints = [np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0]), np.array([-1, 0, 0, 0])]
initial_condition = np.array([0, 0, 0, 0])

# Setpoint change times
setpoint_change_times = [0, 10, 20]

# Define y-labels
ylabels = ["Position", "Velocity", "Angle", "Angular Velocity"]

# Plot the states and control effort
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

for k, lqr in enumerate(lqrs):
    # Container for the states over time
    X = []
    U = []  # store control effort

    # Initialize the state
    x = initial_condition

    # Setpoint index
    sp_index = 0

    # Simulation loop
    for i in range(1, len(T)):
        # Time span for this iteration
        tspan = [T[i-1], T[i]]

        # Update the setpoint if needed
        if sp_index < len(setpoint_change_times) - 1 and T[i] < setpoint_change_times[sp_index+1] <= T[i+1]:
            sp_index += 1

        # Compute the control input
        u = lqr.get_control(x, setpoints[sp_index])

        # Solve for the next state
        x_next = odeint(system_dynamics, x, tspan, args=(u,))[1]  # odeint returns [x(t0), x(t1)]

        # Update the state
        x = x_next

        # Save the state and control effort to the history
        X.append(x)
        U.append(u)

    # Convert lists to numpy arrays
    X = np.array(X)
    U = np.array(U)

    # Plot the states
    for j in range(4):
        axs[j].plot(T[:-1], X[:, j], label=labels[k])
        axs[j].set_ylabel(ylabels[j])
        axs[j].grid(True)

    # Plot the control effort
    axs[4].plot(T[:-1], U, label=labels[k])
    axs[4].set_xlabel('Time [s]')
    axs[4].set_ylabel('Control Effort')
    axs[4].grid(True)

# Add legend to each subplot
for ax in axs:
    ax.legend()

plt.suptitle('LQR Controller States and Control Effort')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate title

plt.savefig("lqr_controller.png")
plt.show()