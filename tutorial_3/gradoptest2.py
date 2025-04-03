import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # We'll use this for the optimizer
from autograd import tensor_jacobian_product
from icecream import ic

from nanophoto.meep_compute_fom import compute_FOM

from gradoptest import pogne_opt, MappingClass
from utils import (smooth_image, save_code, nom_fichier, mirror_upper_y_half,
                   normalise)

"""
Test d'optimisation avec Adam
"""


def meep_gradient(image):
    opt, sim_args = pogne_opt()
    Nx = sim_args['Nx']
    Ny = sim_args['Ny']

    mapping = MappingClass(**sim_args)
    x = mirror_upper_y_half(image)
    f0, g0 = opt([mapping(x, 0.5, 256)])
    f0 = f0/2
    backprop_gradient = tensor_jacobian_product(
        mapping, 0)(x, 0.5, 2, g0[:, 0])
    backprop_gradient = backprop_gradient.reshape(Nx, Ny)
    return f0, backprop_gradient


def optimize_device_adam(initial_A, learning_rate=0.01, num_iterations=200):
    """
    Optimizes the device array A using the Adam optimizer.
    """
    A_optimized = np.copy(initial_A)
    performance_history = []

    # Treat A as a flat vector for the optimizer
    initial_params = A_optimized.flatten()

    def loss_and_gradient(params):
        """
        Loss function (negative of performance for maximization) and gradient.
        In a real scenario, the gradient would come from Meep.
        Here, we use the simulated gradient.
        """
        current_A = params.reshape(initial_A.shape)
        fom, gradient_A = meep_gradient(current_A)
        # For maximization, we want to minimize the negative of the performance
        loss = -fom
        gradient = -gradient_A.flatten()
        return loss, gradient

    # Use the L-BFGS-B optimizer which can handle bounds
    bounds = [(0, 1)] * initial_params.size
    result = minimize(loss_and_gradient, initial_params, method='L-BFGS-B',
                      jac=True, bounds=bounds,
                      options={'maxiter': num_iterations,
                               'ftol': 1e-6, 'gtol': 1e-6})

    optimized_A = result.x.reshape(initial_A.shape)
    final_performance = compute_FOM(optimized_A[:, 90:])
    print(
        f"""Optimization complete. Final Performance (FOM):
        {final_performance:.4f}""")

    return optimized_A, performance_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', default=False)
    args = parser.parse_args()
    global debug
    debug = args.d

    jobid = os.environ['SLURM_JOB_ID'] if debug is False else 'debug'
    # jobid = datetime.now().strftime("%m%d_%H%M")
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        savepath = os.path.join(jobid, os.environ['SLURM_ARRAY_TASK_ID'])
    nom = nom_fichier()
    savepath = os.path.join('runs', nom, jobid)
    save_code(savepath)
    os.makedirs(savepath, exist_ok=True)
    args_dict = vars(args)
    opt, sim_args = pogne_opt()

    ic(savepath)

    # Initialize the device array A
    initial_A = np.random.rand(101, 181)
    initial_A = smooth_image(initial_A)
    initial_A = mirror_upper_y_half(initial_A)
    initial_A = normalise(initial_A)
    ic(initial_A.shape)

    # Run the optimization with Adam (using L-BFGS-B with gradient)
    optimized_A, performance_history = optimize_device_adam(initial_A,
                                                            learning_rate=0.1,
                                                            num_iterations=200)

    # Visualize the initial and optimized device
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(initial_A, cmap='viridis', origin='lower')
    plt.title('Initial Device (A)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(optimized_A, cmap='viridis', origin='lower')
    plt.title('Optimized Device (A)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'final_device.png'))

    plt.figure()
    plt.plot(performance_history)
    plt.xlabel('Iteration (Simplified)')
    plt.ylabel('Performance')
    plt.title('Performance over Optimization')
    plt.grid(True)
    plt.savefig(os.path.join(savepath, 'fom_hist.png'))

