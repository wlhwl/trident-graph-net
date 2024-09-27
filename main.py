"""
Setup dataloaders
"""
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.prometheus import TRIDENT1211
from dataset import get_dataloaders
graph_definition = KNNGraph(detector = TRIDENT1211(), nb_nearest_neighbours=16)
dataloaders = get_dataloaders(graph_definition=graph_definition)

"""
Set up network
"""
from TridentModel import TridentNet, default_net_setting
from graphnet.models.gnn import DynEdge
# backbone = TridentNet(default_net_setting, 'cpu')
# Select backbone
backbone = DynEdge(nb_inputs = graph_definition.nb_outputs,
                  global_pooling_schemes=["min", "max", "mean"])


"""
Setup task and Loss function
"""
from task import get_task
task = get_task(backbone=backbone)


import torch
from MyStandardModel import MyStandardModel
model: MyStandardModel = MyStandardModel(graph_definition = graph_definition,
                      backbone = backbone,
                      tasks = task)

import os

l = dataloaders['test_dataloader']
# only use small set of data
data_num = 50
_, s = torch.utils.data.random_split(l.dataset, [len(l.dataset)-data_num, data_num])
import graphnet.data.dataloader as dataLoader
l = dataLoader.DataLoader(s, batch_size=10, num_workers=2, shuffle=False)
print(l.dataset[0])

print(model.predict(dataloader=l))