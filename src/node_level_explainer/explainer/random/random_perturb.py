import time
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import normalize_adj


class RandomExplainer(Explainer):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        
        self.set_reproducibility()

    
    def compute_loss(self, cf_adj, graph_adj, out, out_original)->float:

        loss_graph_dist = sum(sum(abs(cf_adj - graph_adj))) / 2     
        loss_pred = F.cross_entropy(out, out_original)
        return loss_pred + loss_graph_dist, loss_graph_dist, loss_pred

    def explain_node(self, graph: Data, oracle)->Data:

        # Parameters initialization
        counterfactual = None
        best_loss = np.inf

        # Counterfactual parameters
        num_nodes = graph.adj.shape[0]
        normalized_adj = normalize_adj(graph.adj).to(self.device)
        cf_adj = graph.adj.to(self.device)

        features = graph.x.to(self.device)

        out_original = oracle(features, normalized_adj)
        #pred_orig = torch.argmax(out_original, dim=1)[graph.new_idx]
        
        start = time.time()

        for _ in range(self.cfg.technique.epochs):

                P_e = torch.randint(0, 2, (num_nodes, num_nodes), device=self.device)
                cf_adj = P_e * cf_adj
                cf_adj = normalize_adj(cf_adj)

                out = oracle(features, cf_adj)
                pred_cf = torch.argmax(out, dim=1)
                total_loss, loss_graph_dist, loss_pred = self.compute_loss(cf_adj, graph.adj, out[graph.new_idx], out_original[graph.new_idx])
                
                if (pred_cf[graph.new_idx] == graph.targets[graph.new_idx]) and (total_loss < best_loss):

                    oracle_embedding_repr = oracle.get_embedding_repr(features, cf_adj)
                    embedding_repr = torch.mean(oracle_embedding_repr, dim=0)
                    
                    losses = {"prediction": loss_pred, "graph": loss_graph_dist}
                    counterfactual = Data(x=features, 
                                            adj=cf_adj, 
                                            y=torch.argmax(out_original, dim=1),
                                            sub_index=graph.new_idx,
                                            loss=losses,
                                            x_projection=embedding_repr)
                    
                    best_loss = total_loss
                    
                if time.time() - start > self.cfg.timeout:
                
                    return counterfactual

        return counterfactual
    
    @property
    def name(self):
        
        return "RandomPerturbation" 