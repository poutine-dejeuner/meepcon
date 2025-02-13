import os

import meep as mp
import meep.adjoint as mpa
import numpy as np
import autograd.numpy as npa

from matplotlib import pyplot as plt
from icecream import ic

from utils import double_with_mirror, normalise


class simulation_FOM_computer():
    def __init__(self, half_index_map):
        # ## Basic environment setup
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
        center_wavelength = 1.550

        seed = 240
        np.random.seed(seed)
        mp.verbosity(0)
        # Effective permittivity for a Silicon waveguide with thickness 220nm
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

        # ## Design variable setup
        minimum_length = 0.09  # (μm)
        eta_e = 0.75
        filter_radius = mpa.get_conic_radius_from_eta_e(
            minimum_length, eta_e)  # (μm)
        eta_i = 0.5
        eta_d = 1-eta_e
        # int(4*resolution) # (pixels/μm)
        design_region_resolution = int(resolution)
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
        sim_params = (Nx, Ny, filter_radius, design_region_width,
                      design_region_height, design_region_resolution)

        # --------------------
        self.Nx = Nx
        self.Ny = Ny
        self.Sx = Sx
        self.Sy = Sy
        self.size_x = size_x
        self.size_y = size_y
        self.fcen = fcen
        self.filter_radius = filter_radius
        self.design_region_width = design_region_width
        self.design_region_height = design_region_height
        self.design_region_resolution = design_region_resolution
        self.design_region = design_region
        self.arm_separation = arm_separation

        self.frequencies = 1/np.linspace(1.5, 1.6, 5)

        self.sim = sim

        self.monsize = mp.Vector3(y=3*waveguide_width)
        self.source_center = mp.Vector3(x=source_x + 0.1)
        self.topmoncenter = mp.Vector3(size_x/2, arm_separation, 0)
        self.abs_src_coeff = 57.97435797757672

        self.update_design_region_params(half_index_map)
        self.sim_has_run = False

    def double_with_mirror(image):
        channels = '~/scratch/nanophoto/lowfom/nodata/fields/channels.npy'
        channels = np.load(os.path.expanduser(channels))
        mirrored_image = np.fliplr(image)  # Crée l'image miroir
        doubled_image = np.concatenate((mirrored_image[:, :-1], image), axis=1)
        return doubled_image

    def normalise(image):
        image = (image - image.min()) / (image.max() - image.min())
        return image

    def mapping(self, x, eta, beta):
        x = (npa.fliplr(x.reshape(self.Nx, self.Ny)) +
             x.reshape(self.Nx, self.Ny))/2  # up-down symmetry
        # filter
        filtered_field = mpa.conic_filter(x, self.filter_radius,
                                          self.design_region_width,
                                          self.design_region_height,
                                          self.design_region_resolution)
        # projection
        projected_field = mpa.tanh_projection(filtered_field, beta, eta)
        # interpolate to actual materials
        return projected_field.flatten()

    def update_design_region_params(self, image):
        self.sim.reset_meep()
        idx_map = double_with_mirror(image)
        idx_map = normalise(idx_map)
        index_map = self.mapping(idx_map, 0.5, 256)
        self.design_region.update_design_parameters(index_map)
        self.sim.run(until_after_sources=100)
        return

    def compute_dft_Ex_fields(self):
        # full field monitor
        size = mp.Vector3(self.Sx, self.Sy, 0)
        dft_monitor = self.sim.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez],             # Components to monitor
            self.fcen, 0, 1,
            center=mp.Vector3(0, 0, 0),        # Center of the monitor region
            size=self.monsize          # Size of the monitor region
        )
        self.sim.run(until_after_sources=100)
        meepfx = self.sim.get_dft_array(dft_monitor, mp.Ex, 0)
        meepfx = np.real(meepfx)
        ic(meepfx.shape)
        plt.imshow(meepfx)
        plt.savefig('dft_field.png')
        # plt.axis('off')
        # plt.show()

    def get_eigenmode_coeffs(self, sim, fluxregion):
        sim.reset_meep()
        flux = sim.add_flux(self.fcen, 0, 1, fluxregion)
        # breakpoint()
        mon_pt = mp.Vector3(*self.source_center)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez,
                                                                mon_pt, 1e-9))
        # res = sim.get_eigenmode_coefficients(flux, [1])
        res = sim.get_eigenmode_coefficients(flux, [1],
                                             eig_parity=mp.EVEN_Z+mp.ODD_Y)
        coeffs = res.alpha
        return coeffs

    def compute_fom(self):
        # Get top output flux coefficients
        topmoncenter = mp.Vector3(self.size_x/2, self.arm_separation, 0)
        topfluxregion = mp.FluxRegion(self.topmoncenter, self.monsize)
        top_coeffs = self.get_eigenmode_coeffs(self.sim, topfluxregion)
        # fom1 = np.abs(top_coeffs[0, 0, 0])**2/np.abs(src_coeffs[0, 0, 0])**2
        fom = np.abs(top_coeffs[0, 0, 0])**2/self.abs_src_coeff
        return fom

    def compute_gradient(self):
        import meep.adjoint as mpa

        mode = 1
        center = self.topmoncenter
        size = self.monsize
        volume = mp.Volume(center=center, size=size)
        ob_list = [mpa.EigenmodeCoefficient(self.sim, volume, mode)]

        def J(top):
            return npa.mean(npa.abs(top)**2)

        opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=J,
            objective_arguments=ob_list,
            design_regions=[self.design_region],
            frequencies=self.frequencies
        )
        # opt.plot2D(True)
        x0 = 0.5*np.ones((self.Nx, self.Ny))
        f0, g0 = opt([self.mapping(x0, 0.5, 2)])

        plt.figure()
        plt.imshow(np.rot90(g0[:, 0].reshape(self.Nx, self.Ny)))
        plt.colorbar()
        plt.savefig('gradient.png')

        return


if __name__ == '__main__':
    # TODO checker si tout les sim.run et sim.meep_reset sont en place
    path = '~/scratch/nanophoto/lowfom/nodata/fields/'
    path = os.path.expanduser(path)
    image = np.load(os.path.join(path, 'images.npy'), mmap_mode='r')[0]
    sim = simulation_FOM_computer(image)

    def fom_comp_test(sim):
        fom = sim.compute_fom()
        ic(fom)

    def field_test(sim):
        # BUG
        sim.compute_dft_Ex_fields()

    def gradient_test(sim):
        sim.compute_gradient()

    # fom_comp_test(sim)
    # field_test(sim)
    gradient_test(sim)
