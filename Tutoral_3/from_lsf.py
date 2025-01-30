import os
import numpy as np
import autograd.numpy as npa
from matplotlib import pyplot as plt

import meep as mp
import meep.adjoint as mpa

from icecream import ic
import pint
from utils import double_with_mirror, normalise


def mapping(x, eta, beta):
    # eta in [0,1], skews the distribution towards one material or the other?
    # higher beta makes design more binarized
    # up-down symmetry
    ic(x.shape)
    ic(Nx, Ny)
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2
    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height,
                                      design_region_resolution)
    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    # interpolate to actual materials
    return projected_field.flatten()


ur = pint.UnitRegistry()
seed = 240
np.random.seed(seed)
mp.verbosity(0)
PATH = '~/scratch/nanophoto/lowfom/nodata/fields'
PATH = os.path.expanduser(PATH)


# Lumerical script file
# ## SIM PARAMS
# opt_size_x=3.5e-6;
# opt_size_y=3.5e-6;
# size_x=opt_size_x+0.6e-6;
# size_y=opt_size_y+1e-6;
# out_wg_dist = 1.25e-6;
# wg_width = 0.5e-6;
# mode_width = 3*wg_width;
# wg_index = 2.8;
# bg_index = 1.44;
# dx = 20e-9;

# size_x et size_y sont ecrases par le script python
# opt_size_x = 3.5e-6*ur.meter
# opt_size_y = 3.5e-6*ur.meter
dx = 20e-9*ur.meter/ur.pixel
opt_size_x = 2000e-9*ur.meter + dx*ur.pixel
opt_size_y = 2*1800e-9*ur.meter + dx*ur.pixel
size_x = opt_size_x+0.6e-6*ur.meter
size_y = opt_size_y+1e-6*ur.meter
out_wg_dist = 1.25e-6*ur.meter
wg_width = 0.5e-6*ur.meter
mode_width = 3*wg_width
wg_index = 2.8
bg_index = 1.44
wg_zspan = 220e-9*ur.meter

opt_xpixel = opt_size_x*(1/dx)
opt_ypixel = opt_size_y*(1/dx)
ic(opt_xpixel.magnitude, opt_ypixel.magnitude)

# ## GEOMETRY
# #INPUT WAVEGUIDE
# addrect;
# set('name','input wg');
# set('x min',-size_x);
# set('x max',-opt_size_x/2 + 1e-7);
# set('y',0);
# set('y span',wg_width);
# set('z',0);
# set('z span',220e-9);
# set('index',wg_index);
source_wg_xmin = -size_x
source_wg_xmax = -opt_size_x/2 + 1e-7*ur.meter
source_wg_y = 0
source_wg_yspan = wg_width
source_wg_z = 0
source_wg_zspan = wg_zspan

# ## OUTPUT WAVEGUIDES
# addrect;
# set('name','output wg top');
# set('x min',opt_size_x/2 - 1e-7);
# set('x max',size_x);
# set('y',out_wg_dist);
# set('y span',wg_width);
# set('z',0);
# set('z span',220e-9);
# set('index',wg_index);
top_wg_xmin = opt_size_x/2 - 1e-7*ur.meter
ic(top_wg_xmin)
top_wg_xmax = size_x
top_wg_y = out_wg_dist
top_wg_yspan = wg_width
top_wg_z = 0
top_wg_zspan = wg_zspan

# addrect;
# set('name','output wg bottom');
# set('x min',opt_size_x/2 - 1e-7);
# set('x max',size_x);
# set('y',-out_wg_dist);
# set('y span',wg_width);
# set('z',0);
# set('z span',220e-9);
# set('index',wg_index);
bot_wg_xmin = top_wg_xmin
bot_wg_xmax = top_wg_xmax
bot_wg_y = -out_wg_dist
bot_wg_yspan = wg_width
bot_wg_z = 0
bot_wg_zspan = wg_zspan


# ## SOURCE
# addmode;
# set('direction','Forward');
# set('injection axis','x-axis');
# set('x',-size_x/2 + 1e-7);
# set('y',0);
# set('y span',mode_width);
# set('z',0);
# set('z span',1e-6);
# set('center wavelength',1550e-9);
# set('wavelength span',0);
# set('mode selection','fundamental TE mode');
source_x = -size_x/2 + 1e-7*ur.meter
source_y = 0
source_yspan = mode_width
source_z = 0
source_zspan = 1e-6*ur.meter
center_wavelength = 1550e-9*ur.meter

# ## FDTD
# addfdtd;
# set('dimension','2D');
# set('background index',bg_index);
# set('mesh accuracy',4);
# set('x min',-size_x/2);
# set('x max',size_x/2);
# set('y min',-size_y/2);
# set('y max',size_y/2);
# set('y min bc','anti-symmetric');
# #set('force symmetric y mesh',1);
# set('auto shutoff min',1e-7);

# ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
# addpower;
# set('name','opt_fields');
# set('monitor type','2D Z-normal');
# set('x',0);
# set('x span',opt_size_x);
# set('y min',0);
# set('y max',opt_size_y/2);

# ## FOM FIELDS
# addpower;
# set('name','fom');
# set('monitor type','2D X-normal');
# set('x',size_x/2 - 1e-7);
# set('y',out_wg_dist);
# set('y span',mode_width);
# addmesh;
# set('name','fom_mesh');
# set('override x mesh',true);
# set('dx',dx);
# set('override y mesh',false);
# set('override z mesh',false);
# set('x',size_x/2 - 1e-7);
# set('x span',2*dx);
# set('y',out_wg_dist);
# set('y span',mode_width);


# Effective permittivity for a Silicon waveguide with a thickness of 220nm
Si = mp.Medium(index=wg_index)
SiO2 = mp.Medium(index=bg_index)
# (0.02)*ur.micrometers  size of a pixel (in μm) 20 nm in lumerical exp
delta = dx.to(ur.micrometers/ur.pixel)
# resolution = 20 # (pixels/μm)
resolution = 1/delta  # pixels/μm
waveguide_width = (wg_width).to(ur.micrometer)  # 0.5 # (μm)

# The above settings are not compatible with the (101, 91) design shapes of the
# previous experiment. the values below replace the optimisation regions with
# compatible shapes
# design_shape= (101, 182)
# design_sizex = design_shape[0]*ur.pixel*dx
# design_sizey = design_shape[1]*ur.pixel*dx
design_region_width = opt_size_x.to(ur.micrometer)  # (μm)
design_region_height = opt_size_y.to(ur.micrometer)  # (μm)
# design_region_width = design_sizex.to(ur.micrometer)  # (μm)
# design_region_height = design_sizey.to(ur.micrometer)  # (μm)

# 1.0  (μm) distance between arms center to center
arm_separation = (out_wg_dist).to(ur.micrometer)

# ic((design_region_height*resolution).magnitude)
# ic(design_region_width*resolution)
# test = (int((design_region_width*resolution).magnitude),
#         int((design_region_height*resolution).magnitude))
# ic(test)
# assert test == design_shape

waveguide_length = (
    source_wg_xmax - source_wg_xmin).to(ur.micrometer)  # 0.5  (μm)
pml_size = 1.0  # (μm)

ic(resolution, waveguide_length, waveguide_width,
   design_region_height, design_region_width, arm_separation)

delta = delta.magnitude
resolution = resolution.magnitude
waveguide_width = waveguide_width.magnitude
design_region_width = design_region_width.magnitude
design_region_height = design_region_height.magnitude
arm_separation = arm_separation.magnitude
waveguide_length = waveguide_length.magnitude
source_yspan = source_yspan.magnitude


# ## Design variable setup

minimum_length = 0.09  # (μm)
eta_e = 0.75
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)  # (μm)
eta_i = 0.5
eta_d = 1-eta_e
design_region_resolution = int(resolution)  # int(4*resolution) # (pixels/μm)
frequencies = 1/np.linspace(1.5, 1.6, 5)  # (1/μm)

Nx = round(design_region_resolution*design_region_width)
Ny = round(design_region_resolution*design_region_height)

size = mp.Vector3(design_region_width, design_region_height)
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si)
design_region = mpa.DesignRegion(design_variables,
                                 volume=mp.Volume(center=mp.Vector3(),
                                                  size=size))


# ## Simulation Setup

Sx = 2*pml_size + waveguide_length + design_region_width  # cell size in X
Sy = 2*pml_size + design_region_height + 0.5  # cell size in Y
cell_size = mp.Vector3(Sx, Sy)

pml_layers = [mp.PML(pml_size)]

fcen = (1/center_wavelength.to(ur.micrometer)).magnitude  # 1/1.55
width = 0.2
fwidth = width * fcen
source_center = [-Sx/2 + pml_size + waveguide_length/3, 0, 0]
source_size = mp.Vector3(0, source_yspan, 0)  # mp.Vector3(0,2,0)
kpoint = mp.Vector3(1, 0, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.EigenModeSource(src,
                             eig_band=1,
                             direction=mp.NO_DIRECTION,
                             eig_kpoint=kpoint,
                             size=source_size,
                             center=source_center,
                             eig_parity=mp.EVEN_Z+mp.ODD_Y)]

geometry = [
    # left waveguide
    mp.Block(center=mp.Vector3(x=-Sx/4), material=Si,
             size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
    # top right waveguide
    mp.Block(center=mp.Vector3(x=Sx/4, y=arm_separation/2), material=Si,
             size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
    # bottom right waveguide
    mp.Block(center=mp.Vector3(x=Sx/4, y=-arm_separation/2), material=Si,
             size=mp.Vector3(Sx/2+1, waveguide_width, 0)),
    mp.Block(center=design_region.center,
             size=design_region.size, material=design_variables)
]

# index map from Lumerical settings
image_idx = 1
image = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')[image_idx]
idx_map = double_with_mirror(image)
idx_map = normalise(idx_map)

# random index map
# idx_map = np.random.rand(Nx,Ny)

index_map = mapping(idx_map, 0.5, 256)
design_region.update_design_parameters(index_map)

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    default_material=SiO2,
                    resolution=resolution)

# sim.plot2D()
# plt.show()


def get_sim_coeffs_from_flux_region(sim, fluxregion):
    sim.reset_meep()
    flux = sim.add_flux(fcen, 0, 1, fluxregion)
    # breakpoint()
    mon_pt = mp.Vector3(*source_center)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt,
                                                            1e-9))
    accumulated_flux_spectrum = mp.get_fluxes(flux)
    return accumulated_flux_spectrum


# Get incident flux coefficients
# source_center = [-Sx/2 + pml_size + waveguide_length/3, 0, 0]
mon_pt = mp.Vector3(x=-Sx/2 + pml_size + 2*waveguide_length/3)
monsize = mp.Vector3(y=3*waveguide_width)
fluxregion = mp.FluxRegion(center=mon_pt, size=monsize)
source_flux_spectrum = get_sim_coeffs_from_flux_region(sim, fluxregion)
# sim.plot2D()
# plt.show()

# Get top output flux coefficients
topmoncenter = mp.Vector3(
    Sx/2 - pml_size - 2*waveguide_length/3, arm_separation/2, 0)
topfluxregion = mp.FluxRegion(topmoncenter, monsize)
top_flux_spectrum = get_sim_coeffs_from_flux_region(sim, topfluxregion)
# sim.plot2D()
# plt.show()

FOM2 = (top_flux_spectrum[0])/source_flux_spectrum[0]
ic(FOM2)
ic(np.load(os.path.join(PATH, 'fom.npy'), mmap_mode='r')[image_idx])

_, axes = plt.subplots(1, 3)
sim.plot2D(fields=mp.Ex, ax=axes[0])
axes[0].set_title('Ex')
sim.plot2D(fields=mp.Ey, ax=axes[1])
axes[1].set_title('Ey')
sim.plot2D(fields=mp.Hz, ax=axes[2])
axes[2].set_title('Hz')
plt.save('meepfields.npy')
# plt.show()
