import time
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from ....abstract.explainer import Explainer
from ...utils.utils import normalize_adj

class RandomFeaturesExplainer(Explainer):
    
    def __init__(self, cfg:DictConfig) -> None:
        super().__init__(cfg=cfg)

        self.set_reproducibility()


    def explain_node(self, graph: Data, oracle)->dict:

        norm_adj = normalize_adj(graph.adj)
        features = graph.x.to(self.device)

        out_original = oracle(features, norm_adj)

        best_loss = np.inf
        counterfactual = None
        
        start = time.time()
        
        for _ in range(self.cfg.technique.epochs):
            
            P_x = torch.randint_like(input=features, low=0, high=2, device=self.device)
            perturbed_features = features * P_x

            out = oracle(perturbed_features, norm_adj)
            pred_cf = torch.argmax(out, dim=1)


            loss_pred = F.cross_entropy(out[graph.new_idx], out_original[graph.new_idx])
            loss_feat = F.l1_loss(features, perturbed_features)

            loss_tot = loss_feat + loss_pred
                
            if (pred_cf[graph.new_idx] == graph.targets[graph.new_idx]) and (loss_tot < best_loss):

                embedding_repr = torch.mean(oracle.get_embedding_repr(perturbed_features, norm_adj), dim=0)
                losses = {"prediction": loss_pred, "features": loss_feat}
                

                counterfactual = Data(x=perturbed_features, 
                                            adj=graph.adj, 
                                            y=torch.argmax(out_original, dim=1),
                                            sub_index=graph.new_idx,
                                            loss=losses,
                                            x_projection=embedding_repr)

                best_loss = loss_feat

            if time.time() - start > self.cfg.timeout:
                
                return counterfactual
            
        return counterfactual

    @property
    def name(self):
        
        return "RandomFeatures" 