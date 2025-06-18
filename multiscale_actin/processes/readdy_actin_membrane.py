import numpy as np

from process_bigraph import Process, Composite, gather_emitter_results
from process_bigraph.emitter import add_emitter_to_composite
from process_bigraph.process_types import ProcessTypes

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
        'add_monomer_box_potentials': 'boolean',
        "use_box_actin": 'boolean',
        "actin_box_center_x": 'float',
        "actin_box_center_y": 'float',
        "actin_box_center_z": 'float',
        "actin_box_size_x": 'float',
        "actin_box_size_y": 'float',
        "actin_box_size_z": 'float',
        'add_extra_box': 'boolean',
        'barbed_binding_site': 'boolean',
        'binding_site_reaction_distance': 'float',
    }

    def initialize(self, config):
        actin_simulation = ActinSimulation(self.config)
        self.readdy_system = actin_simulation.system
        self.readdy_simulation = actin_simulation.simulation

    def inputs(self):
        return {
            'topologies': 'map[topology]',
            'particles': 'map[particle]',
        }

    def outputs(self):
        return {
            'topologies': {
                '_type': 'map[topology]',
                '_apply': 'set',
            },
            'particles': {
                '_type': 'map[particle]',
                '_apply': 'set',
            },
        }

    def update(self, inputs, interval):
        ReaddyUtil.add_monomers_from_data(self.readdy_simulation, inputs)

        simulate_readdy(
            self.config["internal_timestep"], 
            self.readdy_system, 
            self.readdy_simulation, 
            interval
        )

        readdy_monomers = ReaddyUtil.get_current_monomers(
            self.readdy_simulation.current_topologies
        )
        transformed_monomers = transform_monomers(
            readdy_monomers, self.config["box_center"]
        )

        return transformed_monomers


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
        observe = readdy_actions.evaluate_observables()
        init()
        create_nl()
        calculate_forces()
        update_nl()
        observe(0)
        n_steps = int(timestep * 1e9 / internal_timestep)
        for t in range(1, n_steps + 1):
            diffuse()
            update_nl()
            react()
            update_nl()
            calculate_forces()
            observe(t) # do we need this?

    readdy_simulation._run_custom_loop(loop, show_summary=False)


def transform_monomers(monomers, box_center):
    for particle_id in monomers["particles"]:
        monomers["particles"][particle_id]["position"] += box_center
    return monomers


def run_readdy_actin_membrane(total_time=1e-9):
    config = {
        "name": "actin_membrane",
        "timestep": 1.0,
        "internal_timestep": 0.1,  # ns
        "box_size": np.array([float(150.0)] * 3),  # nm
        "periodic_boundary": True,
        "reaction_distance": 1.0,  # nm
        "n_cpu": 4,
        "only_linear_actin_constraints": True,
        "reactions": True,
        "dimerize_rate": 1e-30,  # 1/ns
        "dimerize_reverse_rate": 1.4e-9,  # 1/ns
        "trimerize_rate": 2.1e-2,  # 1/ns
        "trimerize_reverse_rate": 1.4e-9,  # 1/ns
        "pointed_growth_ATP_rate": 2.4e-5,  # 1/ns
        "pointed_growth_ADP_rate": 2.95e-6,  # 1/ns
        "pointed_shrink_ATP_rate": 8.0e-10,  # 1/ns
        "pointed_shrink_ADP_rate": 3.0e-10,  # 1/ns
        "barbed_growth_ATP_rate": 1e30,  # 1/ns
        "barbed_growth_ADP_rate": 7.0e-5,  # 1/ns
        "nucleate_ATP_rate": 2.1e-2,  # 1/ns
        "nucleate_ADP_rate": 7.0e-5,  # 1/ns
        "barbed_shrink_ATP_rate": 1.4e-9,  # 1/ns
        "barbed_shrink_ADP_rate": 8.0e-9,  # 1/ns
        "arp_bind_ATP_rate": 2.1e-2,  # 1/ns
        "arp_bind_ADP_rate": 7.0e-5,  # 1/ns
        "arp_unbind_ATP_rate": 1.4e-9,  # 1/ns
        "arp_unbind_ADP_rate": 8.0e-9,  # 1/ns
        "barbed_growth_branch_ATP_rate": 2.1e-2,  # 1/ns
        "barbed_growth_branch_ADP_rate": 7.0e-5,  # 1/ns
        "debranching_ATP_rate": 1.4e-9,  # 1/ns
        "debranching_ADP_rate": 7.0e-5,  # 1/ns
        "cap_bind_rate": 2.1e-2,  # 1/ns
        "cap_unbind_rate": 1.4e-9,  # 1/ns
        "hydrolysis_actin_rate": 1e-30,  # 1/ns
        "hydrolysis_arp_rate": 3.5e-5,  # 1/ns
        "nucleotide_exchange_actin_rate": 1e-5,  # 1/ns
        "nucleotide_exchange_arp_rate": 1e-5,  # 1/ns
        "verbose": False,
        "use_box_actin": True,
        "use_box_arp": False,
        "use_box_cap": False,
        "obstacle_radius": 0.0,
        "obstacle_diff_coeff": 0.0,
        "use_box_obstacle": False,
        "position_obstacle_stride": 0,
        "displace_pointed_end_tangent": False,
        "displace_pointed_end_radial": False,
        "tangent_displacement_nm": 0.0,
        "radial_displacement_radius_nm": 0.0,
        "radial_displacement_angle_deg": 0.0,
        "longitudinal_bonds": True,
        "displace_stride": 1,
        "bonds_force_multiplier": 0.2,
        "angles_force_constant": 1000.0,
        "dihedrals_force_constant": 1000.0,
        "actin_constraints": True,
        "add_monomer_box_potentials": False,
        "use_box_actin": False,
        "actin_box_center_x": 12.0,
        "actin_box_center_y": 0.0,
        "actin_box_center_z": 0.0,
        "actin_box_size_x": 20.0,
        "actin_box_size_y": 50.0,
        "actin_box_size_z": 50.0,
        "add_extra_box": False,
        "barbed_binding_site": True,
        "binding_site_reaction_distance": 3.0,
    }

    # make the simulation
    membrane_monomers = get_membrane_monomers(
        center=np.array([25.0, 0.0, 0.0]), 
        size=np.array([0.0, 100.0, 100.0]), 
        particle_radius=2.5,
    )

    actin_monomers = ActinGenerator.get_monomers(
        fibers_data=[
            FiberData(
                28,
                [
                    np.array([-25, 0, 0]),
                    np.array([25, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ], 
        use_uuids=False, 
        start_normal=np.array([0., 1., 0.]), 
        longitudinal_bonds=True,
        barbed_binding_site=True,
    )
    actin_monomers = ActinGenerator.setup_fixed_monomers(
        actin_monomers, 
        orthogonal_seed=True, 
        n_fixed_monomers_pointed=3, 
        n_fixed_monomers_barbed=0,
    )

    state  = {
        'readdy': {
            '_type': 'process',
            'address': 'local:readdy',
            'config': config,
            'inputs': {
                'particles': ['particles'],
                'topologies': ['topologies']        
            },
            'outputs': {
                'particles': ['particles'],
                'topologies': ['topologies']        
            }
        },
        'particles': {},  # put the initial particles here
        'topologies': {},   # put the initial topologies here
    }

    # TODO do this elsewhere?
    core = ProcessTypes()
    particle = {
        'type_name': 'string',
        'position': 'tuple[float,float,float]',
        'neighbor_ids': 'list[int]',
    }
    topology = {
        'type_name': 'string',
        'particle_ids': 'list[int]',
    }
    core.register('topology', topology)
    core.register('particle', particle)

    core.register_process('readdy', ReaddyActinMembrane)
    core.register_process('simularium-emitter', SimulariumEmitter)

    sim = Composite({
        "state": state,
        "emitter": {"mode": "all"}
    }, core=core)

    sim = add_emitter_to_composite(
        sim,
        core,
        emitter_mode='all',
        address='local:simularium-emitter'
    )

    # simulate
    print("Simulating...")
    sim.update({}, total_time)

    results = gather_emitter_results(sim)

    import ipdb; ipdb.set_trace()
    

if __name__ == "__main__":
    run_readdy_actin_membrane()