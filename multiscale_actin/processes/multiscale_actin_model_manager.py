import random
import os
import numpy as np

from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import anyize_paths
from process_bigraph.process_types import ProcessTypes

from simularium_readdy_models.common import get_membrane_monomers

from simularium_readdy_models.actin import (
    ActinGenerator,
    FiberData,
)

from multiscale_actin.processes.multiscale_actin_model_settings import MultiscaleActinModelSettings
from multiscale_actin.processes.readdy_actin_membrane import ReaddyActinMembrane, register_types
from multiscale_actin.processes.simularium_emitter import SimulariumEmitter

class MultiscaleActinModelManager:
    @staticmethod
    def run_readdy_actin_membrane(model_settings: MultiscaleActinModelSettings | None = None,
                                  total_time_in_ns: float=3):
        if model_settings is None:
            model_settings = MultiscaleActinModelSettings()
        pb_config, pb_core = MultiscaleActinModelManager.generate_general_model(model_settings.get_config())
        sim = Composite(pb_config, pb_core)
        sim.run(total_time_in_ns)
        emitter_results = gather_emitter_results(sim)
        return emitter_results

    @staticmethod
    def generate_general_model(readdy_actin_config, use_local_protocol=True) -> tuple[dict, ProcessTypes]:
        random.seed(readdy_actin_config['random_seed'])
        np.random.seed(readdy_actin_config['random_seed'])

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
                'config': readdy_actin_config,
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
                'base_name': readdy_actin_config["output_base_name"],
                "output_dir": readdy_actin_config["output_dir_path"]
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
    config, core = MultiscaleActinModelManager.generate_general_model(settings.get_config())
    model = Composite(config, core)
    print("Preparing to run!")
    model.run(total_time)

    results = gather_emitter_results(model)