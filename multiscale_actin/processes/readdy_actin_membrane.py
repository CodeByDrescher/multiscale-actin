import random

import numpy as np

from process_bigraph import Process, Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires
from process_bigraph.process_types import ProcessTypes
from multiscale_actin.processes.multiscale_actin_model import MultiscaleActinModelSettings

from simularium_readdy_models.actin import (
    ActinSimulation,
    ActinGenerator,
    FiberData,
)
from simularium_readdy_models.common import ReaddyUtil, get_membrane_monomers

from simularium_emitter import SimulariumEmitter


class ReaddyActinMembrane(Process):
    '''
    This process runs ReaDDy models with coarse-grained particle 
    actin filaments and membrane patches.
    '''

    config_schema = {
        'name': 'string',
        'internal_timestep': 'float',
        'box_size': 'tuple[float,float,float]',
        'periodic_boundary': 'boolean',
        'reaction_distance': 'float',
        'n_cpu': 'integer',
        'only_linear_actin_constraints': 'boolean',
        'reactions': 'boolean',
        'dimerize_rate': 'float',
        'dimerize_reverse_rate': 'float',
        'trimerize_rate': 'float',
        'trimerize_reverse_rate': 'float',
        'pointed_growth_ATP_rate': 'float',
        'pointed_growth_ADP_rate': 'float',
        'pointed_shrink_ATP_rate': 'float',
        'pointed_shrink_ADP_rate': 'float',
        'barbed_growth_ATP_rate': 'float',
        'barbed_growth_ADP_rate': 'float',
        'nucleate_ATP_rate': 'float',
        'nucleate_ADP_rate': 'float',
        'barbed_shrink_ATP_rate': 'float',
        'barbed_shrink_ADP_rate': 'float',
        'arp_bind_ATP_rate': 'float',
        'arp_bind_ADP_rate': 'float',
        'arp_unbind_ATP_rate': 'float',
        'arp_unbind_ADP_rate': 'float',
        'barbed_growth_branch_ATP_rate': 'float',
        'barbed_growth_branch_ADP_rate': 'float',
        'debranching_ATP_rate': 'float',
        'debranching_ADP_rate': 'float',
        'cap_bind_rate': 'float',
        'cap_unbind_rate': 'float',
        'hydrolysis_actin_rate': 'float',
        'hydrolysis_arp_rate': 'float',
        'nucleotide_exchange_actin_rate': 'float',
        'nucleotide_exchange_arp_rate': 'float',
        'verbose': 'boolean',
        'use_box_actin': 'boolean',
        'use_box_arp': 'boolean',
        'use_box_cap': 'boolean',
        'obstacle_radius': 'float',
        'obstacle_diff_coeff': 'float',
        'use_box_obstacle': 'boolean',
        'position_obstacle_stride': 'integer',
        'displace_pointed_end_tangent': 'boolean',
        'displace_pointed_end_radial': 'boolean',
        'tangent_displacement_nm': 'float',
        'radial_displacement_radius_nm': 'float',
        'radial_displacement_angle_deg': 'float',
        'longitudinal_bonds': 'boolean',
        'displace_stride': 'integer',
        'bonds_force_multiplier': 'float',
        'angles_force_constant': 'float',
        'dihedrals_force_constant': 'float',
        'actin_constraints': 'boolean',
        'use_box_actin': 'boolean',
        'actin_box_center_x': 'float',
        'actin_box_center_y': 'float',
        'actin_box_center_z': 'float',
        'actin_box_size_x': 'float',
        'actin_box_size_y': 'float',
        'actin_box_size_z': 'float',
        'add_extra_box': 'boolean',
        'barbed_binding_site': 'boolean',
        'binding_site_reaction_distance': 'float',
        'add_membrane': 'boolean',
        "membrane_center_x": 'float',
        "membrane_center_y": 'float',
        "membrane_center_z": 'float',
        "membrane_size_x": 'float',
        "membrane_size_y": 'float',
        "membrane_size_z": 'float',
        'membrane_particle_radius': 'float',
        'obstacle_controlled_position_x': 'float',
        'obstacle_controlled_position_y': 'float',
        'obstacle_controlled_position_z': 'float',
        'random_seed': 'integer'
    }

    def initialize(self, config, readdy_system=None):
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        actin_simulation = ActinSimulation(self.config, False, False, readdy_system)
        self.readdy_system = actin_simulation.system
        self.readdy_simulation = actin_simulation.simulation

    def inputs(self):
        return {
            'topologies': 'map[topology]',
            'particles': 'map[particle]',
        }

    def outputs(self):
        return {
            'topologies': 'map[topology]',
            'particles': 'map[particle]',
        }

    def update(self, inputs, interval):

        self.initialize(self.config, self.readdy_system)

        ReaddyUtil.add_monomers_from_data(self.readdy_simulation, inputs)

        simulate_readdy(
            self.config["internal_timestep"], 
            self.readdy_system, 
            self.readdy_simulation, 
            interval
        )

        id_diff = id_difference(self.readdy_simulation.current_topologies)

        readdy_monomers = ReaddyUtil.get_current_monomers(
            self.readdy_simulation.current_topologies, id_diff
        )

        return readdy_monomers


def id_difference(current_topologies):
    """
    Get the first ID coming out of ReaDDy, it should be zero 
    unless Readdy ran multiple times and cached the IDs, which
    would cause Vivarium to create new particles instead of 
    updating existing particles. 

    (This is a HACK needed as long as ReaDDy has this behavior.)
    """
    return current_topologies[0].particles[0].id


def simulate_readdy(internal_timestep, readdy_system, readdy_simulation, timestep):
    """
    Simulate in ReaDDy for the given timestep
    """
    def loop():
        readdy_actions = readdy_simulation._actions
        init = readdy_actions.initialize_kernel()
        diffuse = readdy_actions.integrator_euler_brownian_dynamics(
            internal_timestep
        )
        calculate_forces = readdy_actions.calculate_forces()
        create_nl = readdy_actions.create_neighbor_list(
            readdy_system.calculate_max_cutoff().magnitude
        )
        update_nl = readdy_actions.update_neighbor_list()
        react = readdy_actions.reaction_handler_uncontrolled_approximation(
            internal_timestep
        )
        init()
        create_nl()
        calculate_forces()
        update_nl()
        n_steps = int(timestep / internal_timestep)
        print(f"running readdy for {n_steps} steps")
        for t in range(1, n_steps + 1):
            diffuse()
            update_nl()
            react()
            update_nl()
            calculate_forces()

    readdy_simulation._run_custom_loop(loop, show_summary=False)


def run_readdy_actin_membrane(total_time_in_ns = 3):
    settings = MultiscaleActinModelSettings()
    config, core = settings.generate_general_model()
    sim = Composite(config, core)
    sim.run(total_time_in_ns)
    results = gather_emitter_results(sim)
    

if __name__ == "__main__":
    run_readdy_actin_membrane()