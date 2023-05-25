import numpy as np
from lqr_controller import LQRController
from tqdm import tqdm
import tensorflow as tf
import cv2

# Define system matrices for a simple system
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])

# Define cost matrices
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
R = np.array([[1.0]])

# Create an LQRController object
lqr = LQRController(A, B, Q, R)

# Define current state and reference state
x = np.array([[1.0], [1.0]])
ref = np.array([[0.0], [0.0]])

# Time parameters
dt = 0.001  # Time step
T = 20.0  # Total simulation time
n_steps = int(T / dt)

# Create an OpenCV window
cv2.namedWindow('Reference')

# Create trackbars for each reference point
for j in range(A.shape[0]):
    cv2.createTrackbar(f'Ref_{j}', 'Reference', -100, 100, lambda x: None)

# List to store states over time
writer = tf.summary.create_file_writer('runs')

# Simulation loop
with writer.as_default():
    for i in tqdm(range(n_steps), desc="Simulation Progress"):

        # Update reference points based on trackbar positions
        for j in range(A.shape[0]):
            ref[j, 0] = cv2.getTrackbarPos(f'Ref_{j}', 'Reference') / 10.0  # scale factor to adjust reference point range

        # Get control signal
        u = lqr.get_control(x, ref)

        # Update state based on system dynamics
        x = x + dt * (A @ x + B @ u)

        # Write states to TensorBoard
        for j in range(A.shape[0]):
            tf.summary.scalar(f'{j}/State', x[j, 0], step=i, description="State Value")
            tf.summary.scalar(f'{j}/Reference', ref[j, 0], step=i, description="Reference Signal")
        tf.summary.scalar('control', u[0, 0], step=i)
