import os
import sys
import shutil
import time
import argparse
import yaml
# from tqdm import tqdm

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product

from matplotlib import pyplot as plt
from icecream import ic
# from orion.client import report_objective

from utils import (normalise, smooth_image, nom_fichier,
                   entgrad_genre)
from nanophoto.meep_compute_fom import meep_get_fields, compute_FOM

mp.verbosity.set(0)


def sigmoid(x, a=1.):
    return 1/(1 + np.exp(-a*x))


def sigmoid_differential(x, a=1.):
    sig = sigmoid(a*x)
    return a*sig*(1 - sig)


def measure_time(func):
    t0 = time.process_time()
    res = func()
    t1 = time.process_time()
    t = t1-t0
    ic(t)
    return res


def save_fields(fields, savepath):
    _, axes = plt.subplots(2, 2)
    fields = [np.real(fields[..., 0]), np.imag(fields[..., 0]),
              np.real(fields[..., 1]), np.imag(fields[..., 1])]
    axes = axes.flatten()
    for i in range(4):
        axes[i].imshow(fields[i])
        axes[i].axis('off')
    plt.savefig(os.path.join(savepath, 'fields.png'))
    plt.clf()


def save_fom_seq(fom_sequence, savepath):
    plt.plot(np.stack(fom_sequence)[1:])
    plt.ylim([0, 0.5])
    path = os.path.join(savepath, 'figures/fomcurve.png')
    plt.savefig(path)
    plt.clf()


def save_code(savepath):
    chemin_nouveau_fichier = os.path.join(savepath, 'code.py')
    try:
        chemin_script_original = sys.argv[0]
        if not os.path.exists(chemin_script_original):
            print(
                f"Erreur: Le fichier original '{chemin_script_original}' n'existe pas.")
            return
        shutil.copy2(chemin_script_original, chemin_nouveau_fichier)
        print(
            f"Le script a été sauvegardé avec succès dans '{chemin_nouveau_fichier}'.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de la sauvegarde: {e}")


def stats(x: np.array):
    ic(x.min(), x.max(), x.mean())


def save_img(image, idx, savepath, titre='', nom=''):
    os.makedirs(os.path.join(savepath, 'figures'), exist_ok=True)
    plt.figure()
    plt.imshow(np.rot90(image), vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')
    plt.title(titre)
    path = os.path.join(savepath, f'figures/{nom}{idx}.png')
    plt.savefig(path)
    plt.clf()


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


def get_opt():
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
    # eta_d = 1-eta_e
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
    # mon_pt = mp.Vector3(*source_center)

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
    # source_fluxregion = mp.FluxRegion(center=source_mon_center,
    #                                   size=monsize,
    #                                   weight=-1)
    # top_fluxregion = mp.FluxRegion(center=top_mon_center,
    #                                size=monsize,
    #                                weight=-1)

    # abs_src_coeff = 57.97435797757672

    # Get top output flux coefficients
    topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
    # topfluxregion = mp.FluxRegion(topmoncenter, monsize)

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
        return npa.mean(power)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=ob_list,
        design_regions=[design_region],
        frequencies=frequencies
    )

    sim_args = {"Nx": Nx, "Ny": Ny,
                "filter_radius": filter_radius,
                "design_region_width": design_region_width,
                "design_region_height": design_region_height,
                "design_region_resolution": design_region_resolution,
                "eta_i": eta_i}
    return opt, sim_args


def ascencion_gradient_a_la_main(opt, sim_args, opt_args, savepath):
    Nx = sim_args['Nx']
    Ny = sim_args['Ny']
    mapping = MappingClass(**sim_args)

    lr_fom = opt_args.lr_fom
    lr_ent = opt_args.lr_ent
    fom_phase = opt_args.fom_phase
    ent_phase = opt_args.ent_phase
    sigma = 20
    slope = 1
    num_loops = fom_phase + ent_phase
    x = np.random.rand(Nx, Ny)
    x = smooth_image(x, sigma)
    x = mirror_upper_y_half(x)
    x = normalise(x)
    save_img(sigmoid(x, slope), -1, savepath)

    fom_sequence = [0]
    i = 0

    for i in range(num_loops):
        print(f'iteration {i}')
        t0 = time.process_time()

        f0, g0 = opt([mapping(x, 0.5, 256)])
        f0 = f0/2
        ic(f0)
        fom_sequence.append(f0)
        if np.abs(fom_sequence[-1] - fom_sequence[-2]) < 1e-3:
            slope += 1

        sigmoid_x = sigmoid(x, slope)
        backprop_gradient = tensor_jacobian_product(
            mapping, 0)(sigmoid_x, 0.5, 2, g0[:, 0])
        backprop_gradient = backprop_gradient.reshape(Nx, Ny)
        backprop_gradient = sigmoid_differential(
            backprop_gradient, slope)*backprop_gradient

        stats(backprop_gradient)
        x = x + lr_fom*backprop_gradient
        # if i > fom_phase:
        # if np.abs(fom_sequence[-1] - fom_sequence[-2]) < 1e-3:
        #     print('binarization step')
        #     x = x - lr_ent*entgrad_genre(x)
        x = mirror_upper_y_half(x)

        print('x apres grad step')
        save_img(sigmoid_x, i, savepath, titre=str(np.round(f0, 3)))
        save_fom_seq(fom_sequence, savepath)

        t1 = time.process_time()
        ic(t1-t0)
    fom = compute_FOM(x[:, 90:])
    fields = meep_get_fields(x[:, 90:])
    save_fields(fields, os.path.join(savepath, 'figures'))
    # report_objective(fom, 'FOM')
    ic(f0)
    print(f'FOM final {fom}')
    np.save(os.path.join(savepath, 'fom.npy'), fom_sequence)
    return


def optimisation_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_fom', type=float, default=1,
                        help='learning rate of the FOM gradient')
    parser.add_argument('-lr_ent', type=float, default=0.1,
                        help='entropy-like gradient component')
    parser.add_argument('-fom_phase', type=int, default=10)
    parser.add_argument('-ent_phase', type=int, default=100)
    parser.add_argument('-d', action='store_true', default=False)
    args = parser.parse_args()
    global debug
    debug = args.d
    if debug is True:
        args.fom_phase = args.ent_phase = 1

    jobid = os.environ['SLURM_JOB_ID'] if debug is False else 'debug'
    # jobid = datetime.now().strftime("%m%d_%H%M")
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        savepath = os.path.join(jobid, os.environ['SLURM_ARRAY_TASK_ID'])
    nom = nom_fichier()
    savepath = os.path.join('runs', nom, jobid)
    save_code(savepath)
    os.makedirs(savepath, exist_ok=True)
    args_dict = vars(args)
    opt, sim_args = get_opt()

    ic(savepath)
    ascencion_gradient_a_la_main(opt, sim_args, args, savepath)
    fichier = os.path.join(savepath, 'config.yml')
    with open(fichier, 'w') as f:
        yaml.dump(args_dict, f)


if __name__ == "__main__":
    optimisation_test()
