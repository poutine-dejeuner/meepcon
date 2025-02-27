import os
from tqdm import tqdm
import timeit
import tracemalloc
import multiprocessing

import meep as mp
# print(mp.__version__)
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa

from matplotlib import pyplot as plt
from icecream import ic

from utils import double_with_mirror, normalise


def mesurer_memoire_fonction(func):
    """
    Décore une fonction pour mesurer son utilisation de mémoire.

    Args:
        func (callable): La fonction à mesurer.
    Returns:
        callable: La fonction décorée.
    """

    def wrapper(*args, **kwargs):
        tracemalloc.start()  # Démarrer le suivi de la mémoire
        resultat = func(*args, **kwargs)  # Exécuter la fonction
        courant, pic = tracemalloc.get_traced_memory()  # Obtenir l'utilisation de mémoire
        tracemalloc.stop()  # Arrêter le suivi

        print(f"Mémoire utilisée par {func.__name__}:")
        print(f"  - Courante: {courant / 10**6:.2f} MB")
        print(f"  - Maximale: {pic / 10**6:.2f} MB")
        return resultat

    return wrapper


def compute_FOM_parallele(images):
    images = [images[i] for i in range(images.shape[0])]
    ic(multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_FOM, images)
    return results


def compute_FOM_array(images):
    if images.ndim == 2:
        return compute_FOM(images)
    foms = []
    for image in images:
        foms.append(compute_FOM(image))
    foms = np.stack(foms)
    return foms


# @mesurer_memoire_fonction
# mem max utilisee 11MB
def compute_FOM(image):
    assert image.shape == (101, 91)
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
    # frequencies = 1/np.linspace(1.5, 1.6, 5)  # (1/μm)

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
    # t1 = timeit.default_timer()
    # ic('init sim', t1-t0)

    def mapping(x, eta, beta):

        x = (npa.fliplr(x.reshape(Nx, Ny)) + x.reshape(Nx, Ny))/2  # up-down symmetry

        # filter
        filtered_field = mpa.conic_filter(x, filter_radius, design_region_width,
                                          design_region_height, design_region_resolution)

        # projection
        projected_field = mpa.tanh_projection(filtered_field, beta, eta)

        # interpolate to actual materials
        return projected_field.flatten()

    idx_map = double_with_mirror(image)
    idx_map = normalise(idx_map)
    index_map = mapping(idx_map, 0.5, 256)
    design_region.update_design_parameters(index_map)

    # full field monitor
    size = mp.Vector3(Sx, Sy, 0)
    dft_monitor = sim.add_dft_fields(
        [mp.Ex, mp.Ey, mp.Ez],             # Components to monitor
        fcen, 0, 1,
        # frequency=fcen,                     # Operating frequency
        center=mp.Vector3(0, 0, 0),        # Center of the monitor region
        size=size          # Size of the monitor region
    )

    monsize = mp.Vector3(y=3*waveguide_width)
    source_mon_center = mp.Vector3(x=source_x + 0.1)
    top_mon_center = mp.Vector3(size_x/2, arm_separation, 0)
    source_fluxregion = mp.FluxRegion(center=source_mon_center,
                                      size=monsize,
                                      weight=-1)
    top_fluxregion = mp.FluxRegion(center=top_mon_center,
                                   size=monsize,
                                   weight=-1)

    sim.run(until_after_sources=100)

    def get_eigenmode_coeffs(sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        # res = sim.get_eigenmode_coefficients(flux, [1])
        res = sim.get_eigenmode_coefficients(flux, [1],
                                             eig_parity=mp.EVEN_Z+mp.ODD_Y)
        coeffs = res.alpha
        return coeffs

    def get_flux_spectrum(sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        accumulated_flux_spectrum = mp.get_fluxes(flux)
        return accumulated_flux_spectrum

    def get_coeffs_flux_spec(sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50,
                                                                mp.Ez,
                                                                mon_pt,
                                                                1e-9))
        res = sim.get_eigenmode_coefficients(flux, [1])
        coeffs = res.alpha
        accumulated_flux_spectrum = mp.get_fluxes(flux)
        return coeffs, accumulated_flux_spectrum

    # Get incident flux coefficients
    # source_mon_pt = mp.Vector3(x=source_x + 0.1)
    # monsize = mp.Vector3(y=3*waveguide_width)
    # source_fluxregion = mp.FluxRegion(center=source_mon_pt, size=monsize)
    # src_coeffs = get_eigenmode_coeffs(sim, source_fluxregion)

    # the np.abs(src_coeffs[0,0,0])**2 was previously computed as
    abs_src_coeff = 57.97435797757672

    # Get top output flux coefficients
    topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
    topfluxregion = mp.FluxRegion(topmoncenter, monsize)
    top_coeffs = get_eigenmode_coeffs(sim, topfluxregion)

    # fom1 = np.abs(top_coeffs[0, 0, 0])**2/np.abs(src_coeffs[0, 0, 0])**2
    fom1 = np.abs(top_coeffs[0, 0, 0])**2/abs_src_coeff
    return fom1


def meep_lumerical_comparison_experiment(num_samples=100):
    PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')
    images = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')[:num_samples]
    foms = np.load(os.path.join(PATH, 'fom.npy'))[:num_samples]
    all_fom = []

    for image_idx in tqdm(range(num_samples)):
        # image_idx = 1
        image = images[image_idx]
        fom = compute_FOM(image)
        all_fom.append(fom)

    all_fom = np.stack(all_fom)
    np.save('meepfom.npy', all_fom)
    err = np.abs(foms - all_fom)
    _, axes = plt.subplots(1, 1)
    axes[0].hist(err, bins=10)
    plt.savefig(f'{image_idx}.png')


def compute_FOM_and_gradients(sim, top_eigenmode_mon):
    import autograd.numpy as npa
    mode = 1
    # TE0 = mpa.EigenmodeCoefficient(sim,
    #         mp.Volume(center=mp.Vector3(x=-Sx/2 + pml_size + 2*waveguide_length/3),
    #             size=mp.Vector3(y=1.5)),mode)
    # TE_top = mpa.EigenmodeCoefficient(sim,
    #         mp.Volume(center=mp.Vector3(Sx/2 - pml_size - 2*waveguide_length/3,arm_separation/2,0),
    #             size=mp.Vector3(y=arm_separation)),mode)
    # TE_bottom = mpa.EigenmodeCoefficient(sim,
    #         mp.Volume(center=mp.Vector3(Sx/2 - pml_size - 2*waveguide_length/3,-arm_separation/2,0),
    #             size=mp.Vector3(y=arm_separation)),mode)
    # ob_list = [TE0,TE_top,TE_bottom]

    # def J(source,top,bottom):
    #     power = npa.abs(top/source) ** 2 + npa.abs(bottom/source) ** 2
    #     return npa.mean(power)
    center = top_eigenmode_mon['center']
    size = top_eigenmode_mon['size']
    volume = mp.Volume(center=center, size=size)
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
    opt.plot2D(True)


if __name__ == '__main__':
    PATH = os.path.expanduser('~/scratch/nanophoto/lowfom/nodata/fields/')

    def test_meep_time():
        image = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')[0]
        t0 = timeit.default_timer()
        fom = compute_FOM(image)
        t1 = timeit.default_timer()
        ic(fom)
        ic(t1-t0)

    def test_parallel_comp():
        num_eval = 2
        images = np.load(os.path.join(PATH, 'images.npy'), mmap_mode='r')
        images = images[:num_eval]
        t0 = timeit.default_timer()
        foms = compute_FOM_parallele(images, num_processes=2)
        t1 = timeit.default_timer()
        ic(t1-t0)
        ic(foms)

    test_parallel_comp()
