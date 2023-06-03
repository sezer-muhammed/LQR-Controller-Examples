import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.integrate import odeint
from control import matlab

# System matrices
A = np.array([[0, 1, 0, 0], [0, -0.1818, 2.6727, 0], [0, 0, 0, 1], [0, -0.4545, 31.1818, 0]])
B = np.array([0, 1.8182, 0, 4.5455]).reshape(-1, 1)
C = np.eye(A.shape[0])
D = np.zeros((A.shape[0], B.shape[1]))

sys = signal.StateSpace(A, B, C, D)

# Q, R matrices for different scenarios
Q_normal = np.eye(A.shape[0])
R_normal = np.eye(B.shape[1])

Q_high = 10 * np.eye(A.shape[0])
R_high = 10 * np.eye(B.shape[1])

scenarios = [
    {"Q": Q_normal, "R": R_normal, "label": "Normal Q, R", "color": "blue"},
    {"Q": Q_high, "R": R_normal, "label": "High Q", "color": "red"},
    {"Q": Q_normal, "R": R_high, "label": "High R", "color": "green"}
]

# Initial condition
x0 = np.array([0.001, 0, 0, 0])

# Time
t = np.arange(0, 5, 0.001)

# Store control efforts for different scenarios
control_efforts = []

for scenario in scenarios:
    # LQR controller
    K, _, _ = matlab.lqr(sys.A, sys.B, scenario["Q"], scenario["R"])

    def system_dynamics(x, t):
        u = -np.dot(K, x)
        return np.squeeze(np.asarray(np.dot(sys.A - sys.B * K, x)))

    # Solve for x
    x = odeint(system_dynamics, x0, t)

    # Calculate control effort
    u = -np.dot(K, x.T)
    control_efforts.append({"u": u, "label": scenario["label"], "color": scenario["color"]})

# Plot control efforts
plt.figure(figsize=(10, 7))

for effort in control_efforts:
    plt.plot(t, effort["u"].T, label=effort["label"], color=effort["color"])

plt.xlabel('Time (sec)')
plt.ylabel('Control Effort')
plt.grid(True)
plt.legend()
plt.title('Control Effort for Different LQR Parameters')
plt.savefig(r'D:\github\LQR Controller\DoubleSidedArticle/simulations/control_effort.png')
