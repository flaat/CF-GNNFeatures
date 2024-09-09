# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.abstract.explainer import Explainer
from ...utils.utils import normalize_adj, get_optimizer
from omegaconf import DictConfig
from torch_geometric.data import Data
from .gcn_perturb import GCNSyntheticPerturb
import time

class CFExplainer(Explainer):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)   
        
        self.set_reproducibility()

    def explain_node(self, graph: Data, oracle):

        self.cf_model = GCNSyntheticPerturb(self.cfg, 
                                      graph.x.shape[1], 
                                      self.cfg.model.hidden, 
                                      self.cfg.model.hidden,
                                      self.num_classes, 
                                      graph.adj, 
                                      self.cfg.model.dropout, 
                                      self.cfg.optimizer.beta)
        
        self.cf_model.load_state_dict(oracle.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

        self.cf_model = self.cf_model.to(self.device)
        self.A_x = graph.adj.to(self.device)
        
        self.D_x = torch.diag(sum(self.A_x))
        self.optimizer = get_optimizer(self.cfg, self.cf_model)
        best_cf_example = None
        self.best_loss = np.inf
        graph = graph.to(self.device)
        start: float = time.time()
        
        for epoch in range(self.cfg.optimizer.num_epochs):
            
            new_sample = self.train(epoch, graph, oracle)

            if time.time() - start > self.cfg.timeout:
                
                return best_cf_example
        
            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example


    def train(self, epoch: int, graph: Data, oracle):
        
        self.optimizer.zero_grad()
        
        
        
        output = self.cf_model.forward(graph.x, graph.adj)      
        output_actual, self.P = self.cf_model.forward_prediction(graph.x)
    
        
        y_non_differentiable = torch.argmax(output_actual[graph.new_idx])
        
        loss, results, cf_adj = self.cf_model.loss(graph, output, y_non_differentiable)
        loss.backward()
       
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        
       
        self.optimizer.step()

        counterfactual = None

        if y_non_differentiable == graph.targets[graph.new_idx] and results["loss_total"] < self.best_loss:
            counterfactual = Data(x=graph.x, 
                                  adj=cf_adj, 
                                  y=torch.argmax(output_actual, dim=1),
                                  sub_index=graph.new_idx,
                                  loss={"prediction": results["loss_pred"],
                                        "graph":results["loss_graph_dist"]},
                                  x_projection=torch.mean(oracle.get_embedding_repr(graph.x, normalize_adj(cf_adj)), dim=0))
            
            self.best_loss = results["loss_total"]

        return counterfactual


    @property
    def name(self):

        return "CF-GNNExplainer"
    