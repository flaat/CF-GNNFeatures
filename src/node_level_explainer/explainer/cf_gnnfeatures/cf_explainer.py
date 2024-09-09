# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import time
from torch_geometric.data import Data
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.node_level_explainer.oracles.perturber.pertuber import NodePerturber
from omegaconf import DictConfig
from ...utils.utils import print_info, get_optimizer
from ....abstract.explainer import Explainer  
from tqdm import tqdm 

class CFExplainerFeatures(Explainer):
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        self.set_reproducibility()


    def explain_node(self, graph: Data, oracle):

        self.best_loss = np.inf
        self.cf_model = NodePerturber(cfg=self.cfg, 
                                      num_classes=self.num_classes, 
                                      adj=graph.adj, 
                                      model=oracle, 
                                      nfeat=graph.x.shape[1], 
                                      nodes_number=graph.x.shape[0]).to(self.device)
        
        self.optimizer = get_optimizer(self.cfg, self.cf_model)
        best_cf_example = None
        
        start = time.time()

        for epoch in range(self.cfg.optimizer.num_epochs):
            
            new_sample = self.train(epoch, graph, oracle)
            
            if time.time() - start > self.cfg.timeout:
                
                return best_cf_example

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example
    

    def train(self, epoch: int, graph: Data, oracle):

        self.optimizer.zero_grad()

        differentiable_output = self.cf_model.forward(graph.x, graph.adj) 
        model_out, V_pert, P_x = self.cf_model.forward_prediction(graph.x) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)

        loss, results, cf_adj = self.cf_model.loss(graph, differentiable_output, y_pred_new_actual[graph.new_idx])
        loss.backward()

        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.optimizer.step()

        counterfactual = None

        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and results["loss_total"] < self.best_loss:
            
            losses = {"feature":results["loss_feat"], "prediction": results["loss_pred"], "graph":results["loss_graph_dist"]}
            embedding_repr = torch.mean(oracle.get_embedding_repr(V_pert, cf_adj), dim=0)

            counterfactual = Data(x=V_pert, 
                                  adj=graph.adj, 
                                  y=y_pred_new_actual,
                                  sub_index=graph.new_idx,
                                  loss=losses,
                                  x_projection=embedding_repr)

            self.best_loss = results["loss_total"]

        return counterfactual
    
    @property
    def name(self):
        
        return "CF-GNNExplainer Features" 
    