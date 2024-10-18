from tensordict import TensorDict
import torch
import numpy as np
import pandas as pd
import networkx as nx
import random

config = {}
edge_list = pd.read_csv('./DATA/skmel2_co_expression_network.csv')
print(edge_list.head())
class GeneSimNetwork():
    def __init__(self, edge_list):
        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        # self.gene_list = gene_list
        # for n in self.gene_list:
        #     if n not in self.G.nodes():
        #         self.G.add_node(n)
        
        # edge_index_ = [(e[0], e[1]) for e in
        #               self.G.edges]

        edge_index_ = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(20)]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        # self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

cell_line_graphs = TensorDict(batch_size = [])
cell_line_weights = TensorDict(batch_size = [])

cell_line_graphs['type1'] = torch.tensor([(1,4),(4,5), (3,1)], dtype=torch.long).T
cell_line_weights['type1'] = torch.Tensor(np.array([1,25,3]))

sim_network = GeneSimNetwork(edge_list)
cell_line_graphs['type2'] = sim_network.edge_index
cell_line_weights['type2'] = sim_network.edge_weight

cell_line_graphs.auto_batch_size_()
cell_line_weights.auto_batch_size_()

print(cell_line_graphs.batch_size)
print(cell_line_weights.batch_size)

config['graphs'] = cell_line_graphs
config['weights'] = cell_line_weights

cell_line_graphs = config['graphs']
cell_line_weights = config['weights']

print(cell_line_graphs['type2'].to('cpu'))
print(cell_line_weights['type2'].to('cpu'))