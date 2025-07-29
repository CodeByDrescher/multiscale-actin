from typing import Any, Dict, Tuple

from process_bigraph import Emitter

import numpy as np
import pandas as pd
from simulariumio import (
    TrajectoryConverter,
    TrajectoryData,
    AgentData,
    MetaData,
    UnitData,
)


class SimulariumEmitter(Emitter):

    def __init__(self, config, core):
        super().__init__(config, core)
        self.configuration_data = None
        self.saved_data: Dict[float, Dict[str, Any]] = {}

    def update(self, state) -> Dict:
        if "particles" in state and "topologies" in state:
            self.saved_data[state['global_time']] = state
        return {}

    def query(self, query=None):
        return {'result' : self.save_simularium_file()}

    def get_simularium_monomers(
        self, time, monomers, actin_radius, position_offset, trajectory
    ):
        """
        Shape monomer state data into Simularium agents
        """
        time_index = len(trajectory["times"])
        trajectory["times"].append(time)
        trajectory["unique_ids"].append([])
        trajectory["type_names"].append([])
        trajectory["positions"].append([])
        edge_ids = []
        edge_positions = []
        for particle_id in monomers.get("particles", {}):
            particle = monomers["particles"][particle_id]
            trajectory["unique_ids"][time_index].append(int(particle_id))
            trajectory["type_names"][time_index].append(particle["type_name"])
            trajectory["positions"][time_index].append(
                np.array(particle["position"]) + position_offset
            )
            # visualize edges between particles
            for neighbor_id in particle["neighbor_ids"]:
                edge = (particle_id, neighbor_id)
                reverse_edge = (neighbor_id, particle_id)
                if edge not in edge_ids and reverse_edge not in edge_ids:
                    edge_ids.append(edge)
                    edge_positions.append(
                        (np.array(particle["position"]) + position_offset).tolist()
                        + (np.array(monomers["particles"][neighbor_id]["position"])
                        + position_offset).tolist()
                    )
        n_agents = len(trajectory["unique_ids"][time_index])
        n_edges = len(edge_ids)
        trajectory["n_agents"].append(n_agents + n_edges)
        trajectory["viz_types"].append(n_agents * [1000.0])
        trajectory["viz_types"][time_index] += n_edges * [1001.0]
        trajectory["unique_ids"][time_index] += [1000 + i for i in range(n_edges)]
        trajectory["type_names"][time_index] += ["edge" for edge in range(n_edges)]
        trajectory["positions"][time_index] += n_edges * [[0.0, 0.0, 0.0]]
        trajectory["radii"].append(n_agents * [actin_radius])
        trajectory["radii"][time_index] += n_edges * [1.0]
        trajectory["n_subpoints"].append(n_agents * [0])
        trajectory["n_subpoints"][time_index] += n_edges * [6]
        trajectory["subpoints"].append(n_agents * [6 * [0.0]])
        trajectory["subpoints"][time_index] += edge_positions
        return trajectory

    @staticmethod
    def fill_df(df, fill):
        """
        Fill Nones in a DataFrame with a fill value
        """
        # Create a dataframe of fill values
        fill_array = [[fill] * df.shape[1]] * df.shape[0]
        fill_df = pd.DataFrame(fill_array)
        # Replace all entries with None with the fill
        df[df.isna()] = fill_df
        return df

    @staticmethod
    def jagged_3d_list_to_numpy_array(jagged_3d_list, length_per_item=3):
        """
        Shape a jagged list with 3 dimensions to a numpy array
        """
        df = SimulariumEmitter.fill_df(pd.DataFrame(jagged_3d_list), length_per_item * [0.0])
        df_t = df.transpose()
        exploded = [df_t[col].explode() for col in list(df_t.columns)]
        result = np.array(exploded).reshape((df.shape[0], df.shape[1], length_per_item))
        return result

    @staticmethod
    def get_agent_data_from_jagged_lists(trajectory, scale_factor) -> AgentData:
        """
        Shape a dictionary of jagged lists into a Simularium AgentData object
        """
        return AgentData(
            times=np.arange(len(trajectory["times"])),
            n_agents=np.array(trajectory["n_agents"]),
            viz_types=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["viz_types"]), 1000.0
            ).to_numpy(),
            unique_ids=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["unique_ids"]), 0
            ).to_numpy(dtype=int),
            types=trajectory["type_names"],
            positions=scale_factor
            * SimulariumEmitter.jagged_3d_list_to_numpy_array(trajectory["positions"]),
            radii=scale_factor
            * SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["radii"]), 0.0
            ).to_numpy(),
            n_subpoints=SimulariumEmitter.fill_df(
                pd.DataFrame(trajectory["n_subpoints"]), 0
            ).to_numpy(dtype=int),
            subpoints=scale_factor
            * SimulariumEmitter.jagged_3d_list_to_numpy_array(trajectory["subpoints"], 6),
        )

    @staticmethod
    def get_simularium_converter(
        trajectory, box_dimensions, scale_factor
    ) -> TrajectoryConverter:
        """
        Shape a dictionary of jagged lists into a Simularium TrajectoryData object
        and provide it to a TrajectoryConverter for conversion
        """
        spatial_units = UnitData("nm")
        spatial_units.multiply(1 / scale_factor)
        return TrajectoryConverter(
            TrajectoryData(
                meta_data=MetaData(
                    box_size=scale_factor * box_dimensions,
                ),
                agent_data=SimulariumEmitter.get_agent_data_from_jagged_lists(
                    trajectory, scale_factor
                ),
                time_units=UnitData("count"),
                spatial_units=spatial_units,
            )
        )

    def save_simularium_file(self) -> str:
        """
        Save the accumulated timeseries history of emitted data to file
        """
        actin_radius = 3.0
        box_dimensions = np.array(3 * [150.])
        trajectory = {
            "times": [],
            "n_agents": [],
            "viz_types": [],
            "unique_ids": [],
            "type_names": [],
            "positions": [],
            "radii": [],
            "n_subpoints": [],
            "subpoints": [],
        }
        for time, state in self.saved_data.items():
            trajectory = self.get_simularium_monomers(
                time,
                state,
                actin_radius,
                np.zeros(3),
                trajectory,
            )
        simularium_converter = SimulariumEmitter.get_simularium_converter(
            trajectory, box_dimensions, 0.1
        )
        output_path = "test"
        simularium_converter.save(output_path)
        return f"saved to {output_path}.simularium"