import os
import time
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product
import nlopt

from matplotlib import pyplot as plt
from icecream import ic
# from orion.client import report_objective

from utils import (normalise, smooth_image,
                   entgrad_genre)
from computeFOM import compute_FOM
from nanophoto.meep_compute_fom import meep_get_fields


mp.verbosity.set(0)


def mirror_upper_y_half(x):
    half = int(x.shape[1]/2)
    upper_half = x[:, half:]
    if x.shape[1] % 2 == 0:
        out = np.concatenate([np.fliplr(upper_half), upper_half], axis=1)
    if x.shape[1] % 2 == 1:
        # si ou a x est de taille impaire en y, on ne repete pas la ligne du
        # milieu
        out = np.concatenate(
            [np.fliplr(upper_half)[:, :-1], upper_half], axis=1)
    return out


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def stats(x: np.array):
    ic(x.min(), x.max(), x.mean())


def save_img(image, idx, savepath):
    os.makedirs(os.path.join(savepath, 'figures'), exist_ok=True)
    plt.figure()
    plt.imshow(np.rot90(image), vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')
    path = os.path.join(savepath, f'figures/opt{idx}.png')
    plt.savefig(path)
    plt.clf()


def get_jobid():
    if debug is True:
        jobid = 'debug'
    elif 'SLURM_JOB_ID' in os.environ: 
        jobid = os.environ['SLURM_JOB_ID']
    else:
        jobid = datetime.now().strftime("%m%d_%H%M")


class MappingClass:
    def __init__(self, **sim_kwargs):
        self.sim_kwargs = sim_kwargs

    def __call__(self, x, eta, beta):
        return mapping(x, eta, beta, **self.sim_kwargs)


def mapping(x, eta, beta, Nx, Ny, filter_radius, design_region_width,
            design_region_height, design_region_resolution, **kwargs):
    # up-down symmetry
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2
    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height,
                                      design_region_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()


def pogne_opt(args):
    pml_size = 1.0  # (μm)

    dx = 0.02
    opt_size_x = 101 * dx
    opt_size_y = 181 * dx
    size_x = 2.6 + pml_size  # um
    size_y = 4.5 + pml_size  # um
    out_wg_dist = 1.25
    wg_width = 0.5
    mode_width = 3*wg_width
    wg_index = 2.8
    bg_index = 1.44

    source_x = -size_x/2 - 0.1
    source_y = 0
    source_yspan = mode_width
    source_z = 0
    # source_zspan = 1
    center_wavelength = 1.550

    # seed = 240
    # np.random.seed(seed)
    mp.verbosity(0)
    # Effective permittivity for a Silicon waveguide with a thickness of 220nm
    Si = mp.Medium(index=wg_index)
    SiO2 = mp.Medium(index=bg_index)
    # size of a pixel (in μm) 20 nm in lumerical exp
    delta = dx
    # resolution = 20 # (pixels/μm)
    resolution = 1/delta  # pixels/μm
    waveguide_width = wg_width  # 0.5 # (μm)
    design_region_width = opt_size_x  # (μm)
    design_region_height = opt_size_y  # (μm)
    # 1.0 (μm) distance between arms center to center
    arm_separation = out_wg_dist
    # waveguide_length = source_wg_xmax - source_wg_xmin  # 0.5 (μm)

    # ## Design variable setup

    minimum_length = 0.09  # (μm)
    eta_e = 0.75
    filter_radius = mpa.get_conic_radius_from_eta_e(
        minimum_length, eta_e)  # (μm)
    eta_i = 0.5
    eta_d = 1-eta_e
    # int(4*resolution) # (pixels/μm)
    design_region_resolution = int(resolution)
    frequencies = 1/np.linspace(1.5, 1.6, 5)  # (1/μm)

    Nx = int(design_region_resolution*design_region_width)
    Ny = int(design_region_resolution*design_region_height)

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si)
    size = mp.Vector3(design_region_width, design_region_height)
    volume = mp.Volume(center=mp.Vector3(), size=size)
    design_region = mpa.DesignRegion(design_variables, volume=volume)

    # ## Simulation Setup

    Sx = 2*pml_size + size_x  # cell size in X
    Sy = 2*pml_size + size_y  # cell size in Y
    cell_size = mp.Vector3(Sx, Sy)

    pml_layers = [mp.PML(pml_size)]

    fcen = 1/center_wavelength  # 1/1.55
    width = 0.2
    fwidth = width * fcen
    source_center = [source_x, source_y, source_z]

    source_size = mp.Vector3(0, source_yspan, 0)
    kpoint = mp.Vector3(1, 0, 0)
    src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
    source = [mp.EigenModeSource(src,
                                 eig_band=1,
                                 direction=mp.NO_DIRECTION,
                                 eig_kpoint=kpoint,
                                 size=source_size,
                                 center=source_center,
                                 eig_parity=mp.EVEN_Z+mp.ODD_Y)]
    mon_pt = mp.Vector3(*source_center)

    geometry = [
        # left waveguide
        mp.Block(center=mp.Vector3(x=-Sx/4),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        # top right waveguide
        mp.Block(center=mp.Vector3(x=Sx/4, y=arm_separation),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        # bottom right waveguide
        mp.Block(center=mp.Vector3(x=Sx/4, y=-arm_separation),
                 material=Si,
                 size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
        mp.Block(center=design_region.center,
                 size=design_region.size,
                 material=design_variables)
    ]

    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=source,
                        symmetries=[mp.Mirror(direction=mp.Y, phase=-1)],
                        default_material=SiO2,
                        resolution=resolution,
                        force_all_components=True)

    size = mp.Vector3(Sx, Sy, 0)
    monsize = mp.Vector3(y=3*waveguide_width)
    source_mon_center = mp.Vector3(x=source_x + 0.1)
    top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    source_fluxregion = mp.FluxRegion(center=source_mon_center,
                                      size=monsize,
                                      weight=-1)
    top_fluxregion = mp.FluxRegion(center=top_mon_center,
                                   size=monsize,
                                   weight=-1)

    abs_src_coeff = 57.97435797757672

    # Get top output flux coefficients
    topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
    topfluxregion = mp.FluxRegion(topmoncenter, monsize)

    mode = 1

    volume = mp.Volume(center=topmoncenter, size=monsize)
    ob_list = [mpa.EigenmodeCoefficient(sim, volume, mode)]

    # -------
    monsize = monsize = mp.Vector3(y=3*waveguide_width)
    source_mon_center = mp.Vector3(x=source_x + 0.1)
    TE0 = mpa.EigenmodeCoefficient(sim,
                                   mp.Volume(center=source_mon_center,
                                             size=monsize), mode)
    top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    TE_top = mpa.EigenmodeCoefficient(sim,
                                      mp.Volume(center=top_mon_center,
                                                size=monsize), mode)

    bot_mon_center = mp.Vector3(size_x/2, -arm_separation, 0)
    TE_bottom = mpa.EigenmodeCoefficient(sim,
                                         mp.Volume(center=bot_mon_center,
                                                   size=monsize), mode)
    ob_list = [TE0, TE_top, TE_bottom]

    # def J(top):
    #     return npa.mean(npa.abs(top)**2)

    def J(source, top, bottom):
        power = npa.abs(top/source) ** 2 + npa.abs(bottom/source) ** 2
        return npa.mean(power)/2

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies
    )

    # PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')
    # image = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')[0]
    # idx_map = double_with_mirror(image)
    # idx_map = normalise(idx_map)
    # # index_map = mapping(idx_map, 0.5, 256)
    # x0 = idx_map
    sim_args = {"Nx": Nx, "Ny": Ny,
                "filter_radius": filter_radius,
                "design_region_width": design_region_width,
                "design_region_height": design_region_height,
                "design_region_resolution": design_region_resolution,
                "eta_i": eta_i}
    return opt, sim_args


def avec_nlopt(opt, sim_args, savepath):
    def f(v, gradient, cur_beta, eta_i):
        print("Current iteration: {}".format(cur_iter[0]+1))

        mapping = MappingClass(**sim_args)
        f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])
        ic(f0)

        plt.figure()
        ax = plt.gca()
        opt.plot2D(False, ax=ax, plot_sources_flag=False,
                   plot_monitors_flag=False, plot_boundaries_flag=False)
        ax.axis('off')
        plt.savefig(os.path.join(savepath, f'{cur_iter[0] + 1}.png'))
        plt.clf()

        if gradient.size > 0:
            gradient[:] = tensor_jacobian_product(mapping, 0)(
                v, eta_i, cur_beta, np.sum(dJ_du, axis=1))

        evaluation_history.append(np.max(np.real(f0)))

        cur_iter[0] = cur_iter[0] + 1

        return np.real(f0)

    Nx = sim_args["Nx"]
    Ny = sim_args["Ny"]
    ic(Nx, Ny)
    eta_i = sim_args["eta_i"]

    evaluation_history = []
    cur_iter = [0]
    algorithm = nlopt.LD_MMA
    n = Nx * Ny  # number of parameters
    # lower and upper bounds
    lb = 0
    ub = 1
    cur_beta = 4
    beta_scale = 2
    num_betas = 6
    update_factor = 12

    # Initial guess
    # x = np.ones((n,)) * 0.5
    x = np.random.rand(Nx, Ny)
    sigma = 20
    x = smooth_image(x, sigma)
    x = normalise(x)
    x = mirror_upper_y_half(x)
    plt.imshow(x)
    plt.savefig('test.png')
    f0, g0 = opt([mapping(x, 0.5, 4, **sim_args)])
    ic(f0)
    plt.figure()
    ax = plt.gca()
    opt.plot2D(False, ax=ax, plot_sources_flag=False,
               plot_monitors_flag=False, plot_boundaries_flag=False)
    plt.savefig(os.path.join(savepath, '0.png'))
    plt.clf()
    x = x.reshape(n)

    for iters in range(num_betas):
        print("current beta: ", cur_beta)

        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta, eta_i))
        solver.set_maxeval(update_factor)
        x[:] = solver.optimize(x)
        cur_beta = cur_beta*beta_scale

    np.save('device.npy', x)
    # ## Final device evaluation
    plt.figure()
    plt.plot(10*np.log10(0.5*np.array(evaluation_history)), 'o-')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Splitting Ratio (dB)')
    plt.savefig(os.path.join(savepath, 'fom_evolution.png'))

    fields = meep_get_fields(x.reshape(Nx, Ny)[:, 90:]) 
    _, axes = plt.subplots(2, 2)
    fields = [np.real(fields[...,0]), np.imag(fields[...,0]), 
              np.real(fields[...,1]), np.imag(fields[...,1])] 
    axes = axes.flatten()
    for i in range(4):
        axes[i].imshow(fields[i])
        axes[i].axis('off')
    plt.savefig(os.path.join(savepath,'fields.png'))


def optimisation_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', default=False)
    args = parser.parse_args()
    global debug
    debug = args.d
    jobid = get_jobid()
    savepath = os.path.join('runs', 'nloptest', jobid)
    ic(savepath)
    os.makedirs(savepath, exist_ok=True)
    args_dict = vars(args)
    opt, sim_args = pogne_opt(args)

    avec_nlopt(opt, sim_args, savepath)


def mirror_upper_half_test():
    im = np.random.rand(190, 205)
    im = smooth_image(im)
    im = im > 1/2

    im1 = mirror_upper_y_half(im)
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(im)
    axes[1].imshow(im1)
    plt.show()


if __name__ == "__main__":
    # mirror_upper_half_test()
    optimisation_test()
