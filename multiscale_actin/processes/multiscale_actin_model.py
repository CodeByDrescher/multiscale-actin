import random

import numpy as np
import os

from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import anyize_paths
from process_bigraph.process_types import ProcessTypes

from simularium_readdy_models.actin import (
    ActinGenerator,
    FiberData,
)
from simularium_readdy_models.common import get_membrane_monomers

from simularium_emitter import SimulariumEmitter

from multiscale_actin.processes.readdy_actin_membrane import ReaddyActinMembrane

def register_types(registry_core: ProcessTypes) -> ProcessTypes:
    if registry_core is None:
        raise ValueError("provided `registry_core` cannot be None!")
    particle = {
        'type_name': 'string',
        'position': 'tuple[float,float,float]',
        'neighbor_ids': 'list[integer]',
        '_apply': 'set',
    }
    topology = {
        'type_name': 'string',
        'particle_ids': 'list[integer]',
        '_apply': 'set',
    }
    registry_core.register('topology', topology)
    registry_core.register('particle', particle)
    return registry_core

class MultiscaleActinModelSettings:
    def __init__(self,
        name: str = "actin_membrane",
        internal_timestep: float = 0.1, # ns
        box_size: np.array = np.array([float(150.0)] * 3), # nm
        periodic_boundary: bool = True,
        reaction_distance: float = 1.0,  # nm
        n_cpu: int = 4,
        only_linear_actin_constraints: bool = True,
        reactions: bool = True,
        dimerize_rate: float = 1e-30,  # 1/ns
        dimerize_reverse_rate: float = 1.4e-9,  # 1/ns
        trimerize_rate: float = 2.1e-2,  # 1/ns
        trimerize_reverse_rate: float = 1.4e-9,  # 1/ns
        pointed_growth_ATP_rate: float = 2.4e-5,  # 1/ns
        pointed_growth_ADP_rate: float = 2.95e-6,  # 1/ns
        pointed_shrink_ATP_rate: float = 8.0e-10,  # 1/ns
        pointed_shrink_ADP_rate: float = 3.0e-10,  # 1/ns
        barbed_growth_ATP_rate: float = 1e30,  # 1/ns
        barbed_growth_ADP_rate: float = 7.0e-5,  # 1/ns
        nucleate_ATP_rate: float = 2.1e-2,  # 1/ns
        nucleate_ADP_rate: float = 7.0e-5,  # 1/ns
        barbed_shrink_ATP_rate: float = 1.4e-9,  # 1/ns
        barbed_shrink_ADP_rate: float = 8.0e-9,  # 1/ns
        arp_bind_ATP_rate: float = 2.1e-2,  # 1/ns
        arp_bind_ADP_rate: float = 7.0e-5,  # 1/ns
        arp_unbind_ATP_rate: float = 1.4e-9,  # 1/ns
        arp_unbind_ADP_rate: float = 8.0e-9,  # 1/ns
        barbed_growth_branch_ATP_rate: float = 2.1e-2,  # 1/ns
        barbed_growth_branch_ADP_rate: float = 7.0e-5,  # 1/ns
        debranching_ATP_rate: float = 1.4e-9,  # 1/ns
        debranching_ADP_rate: float = 7.0e-5,  # 1/ns
        cap_bind_rate: float = 2.1e-2,  # 1/ns
        cap_unbind_rate: float = 1.4e-9,  # 1/ns
        hydrolysis_actin_rate: float = 1e-30,  # 1/ns
        hydrolysis_arp_rate: float = 3.5e-5,  # 1/ns
        nucleotide_exchange_actin_rate: float = 1e-5,  # 1/ns
        nucleotide_exchange_arp_rate: float = 1e-5,  # 1/ns
        verbose: bool = False,
        use_box_actin: bool = True,
        use_box_arp: bool = False,
        use_box_cap: bool = False,
        obstacle_radius: float = 0.0,
        obstacle_diff_coeff: float = 0.0,
        use_box_obstacle: bool = False,
        position_obstacle_stride: int = 0,
        displace_pointed_end_tangent: bool = False,
        displace_pointed_end_radial: bool = False,
        tangent_displacement_nm: float = 0.0,
        radial_displacement_radius_nm: float = 0.0,
        radial_displacement_angle_deg: float = 0.0,
        longitudinal_bonds: bool = True,
        displace_stride: int = 1,
        bonds_force_multiplier: float = 0.2,
        angles_force_constant: float = 1000.0,
        dihedrals_force_constant: float = 1000.0,
        actin_constraints: bool = True,
        actin_box_center_x: float = 12.0,
        actin_box_center_y: float = 0.0,
        actin_box_center_z: float = 0.0,
        actin_box_size_x: float = 20.0,
        actin_box_size_y: float = 50.0,
        actin_box_size_z: float = 50.0,
        add_extra_box: bool = False,
        barbed_binding_site: bool = True,
        binding_site_reaction_distance: float = 3.0,
        add_membrane: bool = True,
        membrane_center_x: float = 25.0,
        membrane_center_y: float = 0.0,
        membrane_center_z: float = 0.0,
        membrane_size_x: float = 0.0,
        membrane_size_y: float = 100.0,
        membrane_size_z: float = 100.0,
        membrane_particle_radius: float = 2.5,
        obstacle_controlled_position_x: float = 0.0,
        obstacle_controlled_position_y: float = 0.0,
        obstacle_controlled_position_z: float = 0.0,
        random_seed: int = 0,
        output_base_name: str = "test",
        output_dir_path: str = ""
    ):
        self.config = {
        "name": name,
        "internal_timestep": internal_timestep,
        "box_size": box_size,
        "periodic_boundary": periodic_boundary,
        "reaction_distance": reaction_distance,
        "n_cpu": n_cpu,
        "only_linear_actin_constraints": only_linear_actin_constraints,
        "reactions": reactions,
        "dimerize_rate": dimerize_rate,
        "dimerize_reverse_rate": dimerize_reverse_rate,
        "trimerize_rate": trimerize_rate,
        "trimerize_reverse_rate": trimerize_reverse_rate,
        "pointed_growth_ATP_rate": pointed_growth_ATP_rate,
        "pointed_growth_ADP_rate": pointed_growth_ADP_rate,
        "pointed_shrink_ATP_rate": pointed_shrink_ATP_rate,
        "pointed_shrink_ADP_rate": pointed_shrink_ADP_rate,
        "barbed_growth_ATP_rate": barbed_growth_ATP_rate,
        "barbed_growth_ADP_rate": barbed_growth_ADP_rate,
        "nucleate_ATP_rate": nucleate_ATP_rate,
        "nucleate_ADP_rate": nucleate_ADP_rate,
        "barbed_shrink_ATP_rate": barbed_shrink_ATP_rate,
        "barbed_shrink_ADP_rate": barbed_shrink_ADP_rate,
        "arp_bind_ATP_rate": arp_bind_ATP_rate,
        "arp_bind_ADP_rate": arp_bind_ADP_rate,
        "arp_unbind_ATP_rate": arp_unbind_ATP_rate,
        "arp_unbind_ADP_rate": arp_unbind_ADP_rate,
        "barbed_growth_branch_ATP_rate": barbed_growth_branch_ATP_rate,
        "barbed_growth_branch_ADP_rate": barbed_growth_branch_ADP_rate,
        "debranching_ATP_rate": debranching_ATP_rate,
        "debranching_ADP_rate": debranching_ADP_rate,
        "cap_bind_rate": cap_bind_rate,
        "cap_unbind_rate": cap_unbind_rate,
        "hydrolysis_actin_rate": hydrolysis_actin_rate,
        "hydrolysis_arp_rate": hydrolysis_arp_rate,
        "nucleotide_exchange_actin_rate": nucleotide_exchange_actin_rate,
        "nucleotide_exchange_arp_rate": nucleotide_exchange_arp_rate,
        "verbose": verbose,
        "use_box_actin": use_box_actin,
        "use_box_arp": use_box_arp,
        "use_box_cap": use_box_cap,
        "obstacle_radius": obstacle_radius,
        "obstacle_diff_coeff": obstacle_diff_coeff,
        "use_box_obstacle": use_box_obstacle,
        "position_obstacle_stride": position_obstacle_stride,
        "displace_pointed_end_tangent": displace_pointed_end_tangent,
        "displace_pointed_end_radial": displace_pointed_end_radial,
        "tangent_displacement_nm": tangent_displacement_nm,
        "radial_displacement_radius_nm": radial_displacement_radius_nm,
        "radial_displacement_angle_deg": radial_displacement_angle_deg,
        "longitudinal_bonds": longitudinal_bonds,
        "displace_stride": displace_stride,
        "bonds_force_multiplier": bonds_force_multiplier,
        "angles_force_constant": angles_force_constant,
        "dihedrals_force_constant": dihedrals_force_constant,
        "actin_constraints": actin_constraints,
        "actin_box_center_x": actin_box_center_x,
        "actin_box_center_y": actin_box_center_y,
        "actin_box_center_z": actin_box_center_z,
        "actin_box_size_x": actin_box_size_x,
        "actin_box_size_y": actin_box_size_y,
        "actin_box_size_z": actin_box_size_z,
        "add_extra_box": add_extra_box,
        "barbed_binding_site": barbed_binding_site,
        "binding_site_reaction_distance": binding_site_reaction_distance,
        "add_membrane": add_membrane,
        "membrane_center_x": membrane_center_x,
        "membrane_center_y": membrane_center_y,
        "membrane_center_z": membrane_center_z,
        "membrane_size_x": membrane_size_x,
        "membrane_size_y": membrane_size_y,
        "membrane_size_z": membrane_size_z,
        'membrane_particle_radius': membrane_particle_radius,
        'obstacle_controlled_position_x': obstacle_controlled_position_x,
        'obstacle_controlled_position_y': obstacle_controlled_position_y,
        'obstacle_controlled_position_z': obstacle_controlled_position_z,
        'random_seed': random_seed,
        'output_base_name': output_base_name,
        'output_dir_path': output_dir_path
    }

    def get_config(self) -> dict:
        return self.config

    def generate_general_model(self, use_local_protocol=True) -> tuple[dict, ProcessTypes]:
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])

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
        membrane_monomers = get_membrane_monomers(
            center=np.array([25.0, 0.0, 0.0]),
            size=np.array([0.0, 100.0, 100.0]),
            particle_radius=2.5,
            start_particle_id=len(actin_monomers["particles"].keys()),
            top_id=1
        )
        free_actin_monomers = ActinGenerator.get_free_actin_monomers(
            concentration=500.0,
            box_center=np.array([12., 0., 0.]),
            box_size=np.array([20., 50., 50.]),
            start_particle_id=len(actin_monomers["particles"].keys()) + len(membrane_monomers["particles"].keys()),
            start_top_id=2
        )
        monomers = {
            'particles': {**actin_monomers['particles'], **membrane_monomers['particles']},
            'topologies': {**actin_monomers['topologies'], **membrane_monomers['topologies']}
        }
        monomers = {
            'particles': {**monomers['particles'], **free_actin_monomers['particles']},
            'topologies': {**monomers['topologies'], **free_actin_monomers['topologies']}
        }

        readdy_address = 'local:readdy' if use_local_protocol \
            else 'python:conda<git+https://github.com/CodeByDrescher/multiscale-actin.git>@multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane'

        state = {
            'readdy': {
                '_type': 'process',
                'address': readdy_address,
                'config': self.config,
                'inputs': {
                    'particles': ['particles'],
                    'topologies': ['topologies']
                },
                'outputs': {
                    'particles': ['particles'],
                    'topologies': ['topologies']
                }
            },
            **monomers
        }

        simularium_address = 'local:simularium-emitter' if use_local_protocol \
            else 'python:conda<git+https://github.com/CodeByDrescher/multiscale-actin.git>@multiscale_actin.processes.simularium_emitter.SimulariumEmitter'

        emitter_wires = {
            'particles': ['particles'],
            'topologies': ['topologies'],
            'global_time': ['global_time']
        }

        state["emitter"] = {
            '_type': 'step',
            'address': simularium_address,
            'config': {
                'emit': anyize_paths(emitter_wires),
                'base_name': self.config["output_base_name"],
                "output_dir": self.config["output_dir_path"]
            },
            'inputs': emitter_wires
        }

        registry_core = register_types(ProcessTypes())

        registry_core.register_process('readdy', ReaddyActinMembrane)
        registry_core.register_process('simularium-emitter', SimulariumEmitter)

        return {"state": state}, registry_core

if __name__ == "__main__":
    print("Creating Multiscale Actin Model!")
    total_time: float = 3.0
    output_base_name = "multiscale_actin"
    output_dir_path = os.path.expanduser("test/output")
    settings = MultiscaleActinModelSettings(output_base_name=output_base_name, output_dir_path=output_dir_path)
    config, core = settings.generate_general_model()
    model = Composite(config, core)
    print("Preparing to run!")
    model.run(total_time)

    results = gather_emitter_results(model)