from typing import Dict, Any
from TRIDENTGraphDefinition import TRIDENT, TRIDENTGraphDefinition
from graphnet.datasets import TRIDENTSmall
from TridentNet import TridentTrackNet, default_net_setting
from MiddleReconModel import MiddleReconModel

config: Dict[str, Any] = {
        "path": "./datasets",
        "batch_size": 3,
        "num_workers": 1,
        "target": "direction",
        "early_stopping_patience":5,
        "fit": {
            "gpus": [0],
            "max_epochs": 200,
        },
    }

features = ['sensor_pos_x','sensor_pos_y','sensor_pos_z', "t"]

graph_definition= TRIDENTGraphDefinition(detector = TRIDENT(),
                                input_feature_names=features)

if __name__ == '__main__':
    data_module = TRIDENTSmall(graph_definition = graph_definition,
                        download_dir = config["path"],
                        train_dataloader_kwargs = {
                            'batch_size': config["batch_size"],
                            'num_workers': config["num_workers"],
                            },
                        backend = 'sqlite')

    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader

    backbone = TridentTrackNet(settings=default_net_setting,DEVICE="cpu")

    batch = next(iter(training_dataloader))

    model = MiddleReconModel(
            backbone=backbone,
            optimizer_kwargs={"lr": 1e-03},
            scheduler_kwargs={
                "patience": 2,
            },
            scheduler_config={
                "frequency": 1,
                "monitor": "val_loss",
            },
        )

    model.fit(train_dataloader=training_dataloader)