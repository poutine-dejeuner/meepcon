import os

import meep as mp
# print(mp.__version__)
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa

from matplotlib import pyplot as plt
from icecream import ic

from utils import double_with_mirror, normalise




# @mesurer_memoire_fonction
# mem max utilisee 11MB

# t0 = timeit.default_timer()
# ## Basic environment setup
pml_size = 1.0  # (μm)

dx = 0.02
opt_size_x = 101 * dx
opt_size_y = 181 * dx
size_x = 2.6 + pml_size  # um
size_y = 4.5 + pml_size  # um
# size_x = opt_size_x + 2*0.4
# size_y = opt_size_y + 2*0.4
out_wg_dist = 1.25
wg_width = 0.5
mode_width = 3*wg_width
wg_index = 2.8
bg_index = 1.44
# wg_zspan = 0.22

# opt_xpixel = opt_size_x*(1/dx)
# opt_ypixel = opt_size_y*(1/dx)

# source_wg_xmin = -size_x
# source_wg_xmax = -opt_size_x/2 + 0.1
# source_wg_y = 0
# source_wg_yspan = wg_width
# source_wg_z = 0
# source_wg_zspan = wg_zspan

# top_wg_xmin = opt_size_x/2 - 0.1
# top_wg_xmax = size_x
# top_wg_y = out_wg_dist
# top_wg_yspan = wg_width
# top_wg_z = 0
# top_wg_zspan = wg_zspan

# bot_wg_xmin = top_wg_xmin
# bot_wg_xmax = top_wg_xmax
# bot_wg_y = -out_wg_dist
# bot_wg_yspan = wg_width
# bot_wg_z = 0
# bot_wg_zspan = wg_zspan

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
                    # symmetries=[mp.Mirror(direction=mp.Y)],
                    default_material=SiO2,
                    resolution=resolution)


# idx_map = double_with_mirror(image)
# idx_map = normalise(idx_map)
# index_map = mapping(idx_map, 0.5, 256)
# design_region.update_design_parameters(index_map)

# full field monitor
size = mp.Vector3(Sx, Sy, 0)
# dft_monitor = sim.add_dft_fields(
#     [mp.Ex, mp.Ey, mp.Ez],             # Components to monitor
#     fcen, 0, 1,
#     # frequency=fcen,                     # Operating frequency
#     center=mp.Vector3(0, 0, 0),        # Center of the monitor region
#     size=size          # Size of the monitor region
# )

monsize = mp.Vector3(y=3*waveguide_width)
source_mon_center = mp.Vector3(x=source_x + 0.1)
top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
source_fluxregion = mp.FluxRegion(center=source_mon_center,
                                  size=monsize,
                                  weight=-1)
top_fluxregion = mp.FluxRegion(center=top_mon_center,
                               size=monsize,
                               weight=-1)

# sim.run(until_after_sources=100)

# the np.abs(src_coeffs[0,0,0])**2 was previously computed as
abs_src_coeff = 57.97435797757672

# Get top output flux coefficients
topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
topfluxregion = mp.FluxRegion(topmoncenter, monsize)

def double_with_mirror(image):
    channels = '~/scratch/nanophoto/lowfom/nodata/fields/channels.npy'
    channels = np.load(os.path.expanduser(channels))
    mirrored_image = np.fliplr(image)  # Crée l'image miroir
    doubled_image = np.concatenate((mirrored_image[:, :-1], image), axis=1)
    return doubled_image


def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image


def mapping(x, eta, beta):
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2  # up-down symmetry
    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height, design_region_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()


# PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')
# image = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')[0]
# idx_map = double_with_mirror(image)
# idx_map = normalise(idx_map)
# index_map = mapping(idx_map, 0.5, 256)

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
# f0, g0 = opt()
# breakpoint()
x0 = 0.5*np.ones((Nx,Ny))
f0, g0 = opt([mapping(x0,0.5,2)])
# ic(g0.shape)
# f0, g0 = opt([index_map])

plt.figure()
print(g0.shape)
plt.imshow(np.rot90(g0[:, 0].reshape(Nx, Ny)))
plt.colorbar()
plt.savefig('grad.png')
