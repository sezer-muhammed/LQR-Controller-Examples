import numpy as np
from scipy.linalg import solve_continuous_are as care
from typing import List, Tuple

class LQRController:
    """
    This class implements a Linear Quadratic Regulator (LQR) controller.

    Attributes:
        A (np.array): The system matrix.
        B (np.array): The control matrix.
        Q (np.array): The state cost matrix.
        R (np.array): The control cost matrix.
        K (np.array): The state feedback gain.
    """

    def __init__(self, A: np.array, B: np.array, Q: np.array, R: np.array):
        """
        The constructor for LQRController class.

        Args:
            A (np.array): The system matrix.
            B (np.array): The control matrix.
            Q (np.array): The state cost matrix.
            R (np.array): The control cost matrix.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()

    def _compute_gain(self) -> np.array:
        """
        Private method to compute the state feedback gain by solving the Algebraic Riccati equation.

        Returns:
            np.array: The state feedback gain.
        """
        # Solve the continuous time algebraic Riccati equation
        P = care(self.A, self.B, self.Q, self.R)

        # Compute the state feedback gain
        K = np.linalg.inv(self.R) @ self.B.T @ P

        return K

    def get_control(self, x: np.array, ref: np.array) -> np.array:
        """
        Method to compute the control signal based on the current state and reference signal.

        Args:
            x (np.array): The current state.
            ref (np.array): The reference signal.

        Returns:
            np.array: The control signal.
        """
        # Compute the state deviation
        x_dev = x - ref

        # Compute the control signal
        u = -self.K @ x_dev

        return u
