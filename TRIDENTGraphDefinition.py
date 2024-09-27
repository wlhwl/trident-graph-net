import numpy as np
from typing import List, Optional, Dict, Any, Callable
import torch
from torch_geometric.data import Data
from graphnet.models.graphs import GraphDefinition
from TRIDENTNodeDefinition import TRIDENTNodeDefinition
from torch_geometric.data import Data
from graphnet.models.detector.detector import Detector
from graphnet.constants import PROMETHEUS_GEOMETRY_TABLE_DIR
import os

class TRIDENT(Detector):
    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "trident.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x# / 1.05e04


class TRIDENTGraphDefinition(GraphDefinition):
    def __init__(
        self,
        detector: TRIDENT = TRIDENT(),
        node_definition: TRIDENTNodeDefinition = TRIDENTNodeDefinition(),
    ) -> None:
        
        super().__init__(
            detector=detector,
            node_definition=node_definition,
            )

    def forward(  # type: ignore
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        truth_dicts: Optional[List[Dict[str, Any]]] = None,
        custom_label_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight: Optional[float] = None,
        loss_weight_default_value: Optional[float] = None,
        data_path: Optional[str] = None,
    ) -> Data:
        
        # print("MyGraphDefinition.forward input_features: ", input_features)
        # Checks
        self._validate_input(
            input_features=input_features,
            input_feature_names=input_feature_names,
        )
        
        # Transform to pytorch tensor
        input_features = torch.tensor(input_features, dtype=self.dtype)

        # Standardize / Scale  node features
        input_features = self._detector(input_features, input_feature_names)
        
        # Create graph & get new node feature names
        graph, node_feature_names = self._node_definition(input_features)

        # Enforce dtype
        graph.x = graph.x.type(self.dtype)

        # Attach number of pulses as static attribute.
        graph.n_pulses = torch.tensor(len(input_features), dtype=torch.int32)

        # Assign edges
        if self._edge_definition is not None:
            graph = self._edge_definition(graph)

        # Attach data path - useful for Ensemble datasets.
        if data_path is not None:
            graph["dataset_path"] = data_path

        # Attach loss weights if they exist
        graph = self._add_loss_weights(
            graph=graph,
            loss_weight=loss_weight,
            loss_weight_column=loss_weight_column,
            loss_weight_default_value=loss_weight_default_value,
        )

        # Attach default truth labels and node truths
        if truth_dicts is not None:
            graph = self._add_truth(graph=graph, truth_dicts=truth_dicts)

        # Attach custom truth labels
        if custom_label_functions is not None:
            graph = self._add_custom_labels(
                graph=graph, custom_label_functions=custom_label_functions
            )

        # Attach node features as seperate fields. MAY NOT CONTAIN 'x'
        graph = self._add_features_individually(
            graph=graph, node_feature_names=node_feature_names
        )

        # Add GraphDefinition Stamp
        graph["graph_definition"] = self.__class__.__name__
        return graph    