import autograd.numpy as npa
import numpy as np
import meep.adjoint as mpa
import meep as mp
from icecream import ic
import matplotlib.pyplot as plt

seed = 240
np.random.seed(seed)
mp.verbosity(0)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

waveguide_width = 0.5  # (μm)
design_region_width = 2.5  # (μm)
design_region_height = 2.5  # (μm)
arm_separation = 1.0  # (μm) distance between arms center to center
waveguide_length = 0.5  # (μm)
pml_size = 1.0  # (μm)
resolution = 20  # (pixels/μm)

minimum_length = 0.09  # (μm)
eta_e = 0.75
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)  # (μm)
eta_i = 0.5
eta_d = 1-eta_e
design_region_resolution = int(4*resolution)  # (pixels/μm)
frequencies = 1/np.linspace(1.5, 1.6, 5)  # (1/μm)

Nx = int(design_region_resolution*design_region_width)
Ny = int(design_region_resolution*design_region_height)

volume = mp.Volume(center=mp.Vector3(),
                   size=mp.Vector3(design_region_width,
                   design_region_height))
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si)
design_region = mpa.DesignRegion(design_variables,
                                 volume=volume)

Sx = 2*pml_size + 2*waveguide_length + design_region_width  # cell size in X
Sy = 2*pml_size + design_region_height + 0.5  # cell size in Y
cell_size = mp.Vector3(Sx, Sy)

pml_layers = [mp.PML(pml_size)]

fcen = 1/1.55
width = 0.2
fwidth = width * fcen
source_center = [-Sx/2 + pml_size + waveguide_length/3, 0, 0]
source_size = mp.Vector3(0, 2, 0)
kpoint = mp.Vector3(1, 0, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.EigenModeSource(src,
                             eig_band=1,
                             direction=mp.NO_DIRECTION,
                             eig_kpoint=kpoint,
                             size=source_size,
                             center=source_center,
                             eig_parity=mp.ODD_Z+mp.EVEN_Y)]

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


def mapping(x, eta, beta):
    """
    input
    x: un array numpy
    eta: ??
    beta: ??

    output
    projecter_field: un objet a utiliser avec design_region.update_design_parameters
    """
    # up-down symmetry
    x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2

    # filter
    filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                      design_region_height,
                                      design_region_resolution)

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    # interpolate to actual materials
    return projected_field.flatten()


def get_sim_coeffs_from_flux_region(sim, fluxregion):
    sim.reset_meep()
    flux = sim.add_flux(fcen, 0, 1, fluxregion)
    # breakpoint()
    mon_pt = mp.Vector3(*source_center)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt,
                                                            1e-9))
    res = sim.get_eigenmode_coefficients(flux, [1],
                                         eig_parity=mp.ODD_Z+mp.EVEN_Y)
    coeffs = res.alpha
    flux_data = mp.get_fluxes(flux)
    return coeffs, flux_data


index_map = mapping(np.random.rand(Nx, Ny), 0.5, 256)
sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    symmetries=[mp.Mirror(direction=mp.Y)],
                    default_material=SiO2,
                    resolution=resolution)
design_region.update_design_parameters(index_map)

# Get incident flux coefficients
# source_center = [-Sx/2 + pml_size + waveguide_length/3, 0, 0]
mon_pt = mp.Vector3(x=-Sx/2 + pml_size + 2*waveguide_length/3)
size = mp.Vector3(y=1.5)
fluxregion = mp.FluxRegion(center=mon_pt, size=size)
source_coeffs, source_flux_data = get_sim_coeffs_from_flux_region(sim, fluxregion)

# Get top output flux coefficients
topmoncenter = mp.Vector3(Sx/2 - pml_size - 2*waveguide_length/3, arm_separation/2, 0)
outmonsize = mp.Vector3(y=arm_separation)
topfluxregion = mp.FluxRegion(topmoncenter, outmonsize)
top_coeffs, top_flux_data = get_sim_coeffs_from_flux_region(sim, topfluxregion)

# Get bot output flux coeffs
botmoncenter = mp.Vector3(Sx/2 - pml_size - 2*waveguide_length / 3,
                    -arm_separation/2, 0)
botfluxregion = mp.FluxRegion(botmoncenter, outmonsize)
bot_mon = sim.add_mode_monitor(fcen, 0, 1, botfluxregion)
bot_coeffs, bot_flux_data = get_sim_coeffs_from_flux_region(sim, botfluxregion)

top_coeff = top_coeffs[0,0,1]
source_coeff = source_coeffs[0,0,0]
bot_coeff = bot_coeffs[0,0,1]
Jt = abs(top_coeff/source_coeff)**2
Jb = abs(bot_coeff/source_coeff)**2
FOM = Jt + Jb
# Ce FOM a un max de 1 plutot que 0.5 comme dans Lumerical. Il faudrait 
# prendre un seul des bras de sortie a monitorer.
ic(FOM)

