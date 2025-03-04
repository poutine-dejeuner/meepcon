import os
import time
import argparse
import yaml

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product

from matplotlib import pyplot as plt
from icecream import ic

from utils import (double_with_mirror, normalise, smooth_image,
                   entgrad_genre)
from calculFOM import compute_FOM


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
    path = os.path.join(savepath,f'figures/opt{idx}.png') 
    plt.savefig(path)
    plt.clf()


parser = argparse.ArgumentParser()
parser.add_argument('-lr_fom', type=float, default=1e17)
parser.add_argument('-lr_ent', type=float, default=0.1)
parser.add_argument('-fom_phase', type=int, default=10)
parser.add_argument('-ent_phase', type=int, default=100)
parser.add_argument('-d', action='store_true', default=False)
args = parser.parse_args()
debug = args.d

jobid = os.environ['SLURM_JOB_ID'] if debug is False else 'debug'
savepath = os.path.join('jobs', jobid)
os.makedirs(savepath, exist_ok=True)
args_dict = vars(args)
fichier = os.path.join(savepath, 'config.yml') 
with open(fichier, 'w') as f:
    yaml.dump(args_dict, f)

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

seed = 240
np.random.seed(seed)
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
arm_separation = out_wg_dist  # 1.0 (μm) distance between arms center to center
# waveguide_length = source_wg_xmax - source_wg_xmin  # 0.5 (μm)

# ## Design variable setup

minimum_length = 0.09  # (μm)
eta_e = 0.75
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)  # (μm)
eta_i = 0.5
eta_d = 1-eta_e
design_region_resolution = int(resolution)  # int(4*resolution) # (pixels/μm)
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
                    symmetries=[mp.Mirror(direction=mp.Y)],
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


def mapping(x, eta, beta):
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2  # up-down symmetry
    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height, design_region_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()

mode = 1

volume = mp.Volume(center=topmoncenter, size=monsize)
ob_list = [mpa.EigenmodeCoefficient(sim, volume, mode)]

def J(top):
    return npa.mean(npa.abs(top)**2)

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


lr_fom = args.lr_fom
lr_ent = args.lr_ent
fom_phase = args.fom_phase
ent_phase = args.ent_phase
if debug is True:
    fom_phase=ent_phase=1
num_loops = fom_phase + ent_phase
# x0 = np.ones((Nx, Ny))*0.5
x0 = np.random.rand(Nx, Ny)
x0 = smooth_image(x0, 20)
save_img(x0, -1, savepath)

fom_sequence = []
for i in range(num_loops):
    t0 = time.process_time()
    print(f"{i}-th optim loop")
    f0, g0 = opt([mapping(x0, 0.5, 256)])
    print('FOM')
    ic(f0)

    fom_sequence.append(f0)
    backprop_gradient = tensor_jacobian_product(mapping,0)(x0,0.5,2,g0[:, 0])
    backprop_gradient = backprop_gradient.reshape(Nx, Ny)
    ic(np.linalg.norm(backprop_gradient))
    print('gradient')
    stats(backprop_gradient)
    x0 = x0 + lr_fom*backprop_gradient 
    if i > fom_phase:
        x0 = x0 - lr_ent*entgrad_genre(x0)
    print('x0 apres grad step')
    stats(x0)
    save_img(x0, i, savepath)

    plt.plot(np.stack(fom_sequence))
    path = os.path.join(savepath, 'figures/fomcurve.png')
    plt.savefig(path)
    t1 = time.process_time()
    ic(t1-t0)
ic(compute_FOM(x0[:, 90:]))
