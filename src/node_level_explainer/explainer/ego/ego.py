# Code taken from TODO: add here
#

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import get_neighbourhood, normalize_adj, print_info
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph, to_dense_adj
from torch_geometric.data import Data


class EgoExplainer(Explainer):
	
    def __init__(self, cfg:DictConfig) -> None:
        super().__init__(cfg=cfg)	

        self.set_reproducibility()
        

    def explain_node(self, graph: Data, oracle) -> dict:

        # Get ego graph = 1hop subgraph
        sub_edge_index = dense_to_sparse(graph.adj)[0]
        ego_nodes, ego_edge_index, _, _ = k_hop_subgraph(graph.new_idx.item(), 1, sub_edge_index)
        ego_adj = to_dense_adj(ego_edge_index, max_num_nodes=graph.adj.shape[0]).squeeze()
        # Get cf_adj, compute prediction for cf_adj
        cf_adj = ego_adj        # keep ego graph
        cf_norm_adj = normalize_adj(cf_adj)

        out = oracle(graph.x, cf_norm_adj)
        out_original = oracle(graph.x, normalize_adj(graph.adj))
        pred_cf = torch.argmax(out, dim=1)[graph.new_idx]
        
        pred_orig = torch.argmax(out_original, dim=1)[graph.new_idx]
        loss_pred = F.cross_entropy(out[graph.new_idx], out_original[graph.new_idx])
        loss_graph_dist = sum(sum(abs(cf_adj - graph.adj))) / 2   

        counterfactual = None
        
            
        if pred_cf != pred_orig:

            embedding_repr = torch.mean(oracle.get_embedding_repr(graph.x, normalize_adj(cf_adj)), dim=0)
            counterfactual = Data(x=graph.x, 
                              adj=cf_adj, 
                              y=torch.argmax(out, dim=1),
                              sub_index=graph.new_idx,
                              loss={"feature": torch.tensor([0.0]),
                                    "prediction": loss_pred,
                                    "graph": loss_graph_dist},
                              x_projection=embedding_repr)
        return counterfactual

    @property
    def name(self):
        
        return "EGO" 