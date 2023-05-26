import numpy as np
from lqr_controller import LQRController
from tqdm import tqdm
import tensorflow as tf

import os
import glob

# Define system matrices for a simple system
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])

# Define cost matrices
Q = np.array([[1.0, 0.0], [0.0, 3.0]])
R = np.array([[1.0]])

# Create an LQRController object
lqr = LQRController(A, B, Q, R)

# Define current state and reference state
x = np.array([[1.0], [1.0]])
ref = np.array([[0.0], [0.0]])
reference_K = np.array([[1/3], [-1/2.5]])

# Time parameters
dt = 0.001  # Time step
T = 20.0  # Total simulation time
n_steps = int(T / dt)


# Get a list of existing directories
existing_directories = glob.glob('runs_*')

if not existing_directories:
    # If there are no existing directories, start with 'runs_1'
    new_dir = 'runs_1'
else:
    # Get the highest existing directory number
    highest_dir_num = max(int(dir.split('_')[-1]) for dir in existing_directories)
    # Increment by 1 for the new directory
    new_dir = f'runs_{highest_dir_num + 1}'
# Create the new directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)

# Now you can use this directory to create your file writer
writer = tf.summary.create_file_writer(new_dir)


# Customizing tqdm parameters
tqdm_kwargs = {
    "desc": "\033[1;36mSimulation Progress\033[0m",  # Bold and colorful description
    "unit": "step",
    "unit_divisor": 1,
    "colour": 'cyan',  # Color of the progress bar
}

# Simulation loop
with writer.as_default():
    for i in tqdm(range(n_steps), **tqdm_kwargs):

        ref = np.array([[min(i * 0.001, 10)], [0.0]])
        ref_input = np.multiply(ref,  reference_K)
        # Get control signal
        u = lqr.get_control(x, ref_input)

        # Update state based on system dynamics
        x = x + dt * (A @ x + B @ u)

        # Write states to TensorBoard
        for j in range(A.shape[0]):
            tf.summary.scalar(f'{j}/State', x[j, 0], step=i, description="State Value")
            tf.summary.scalar(f'{j}/Reference', ref[j, 0], step=i, description="Reference Signal")
        tf.summary.scalar('control', u[0, 0], step=i)
