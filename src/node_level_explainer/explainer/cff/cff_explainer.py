import numpy as np
from omegaconf import DictConfig
import torch
import math
from torch_geometric.data import Data
import time
from src.abstract.explainer import Explainer


class CFFExplainer(Explainer):
    
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.gam = self.cfg.technique.gamma
        self.lam = self.cfg.technique.lam
        self.alp = self.cfg.technique.alpha
        self.epochs = self.cfg.technique.epochs
        self.lr = self.cfg.technique.lr
        
        self.set_reproducibility()
        
    def name(self):
        return "CFF"

    def explain_node(self, graph: Data, oracle) -> dict:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        explainer = ExplainModelNodeMulti(
            graph=graph,
            base_model=oracle
        )
        explainer = explainer.to(device)
            
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.lr, weight_decay=0)
        explainer.train()
        
        best_loss = torch.inf
        counterfactual = None
        
        start_time = time.time()
        
        for _ in range(self.epochs):
            
            explainer.zero_grad()
            
            pred1, pred2 = explainer()
            loss = explainer.loss(pred1[graph.new_idx], pred2[graph.new_idx], graph.y_ground[graph.new_idx], self.gam, self.lam, self.alp)

            loss.backward()
            optimizer.step()
        
            masked_adj = explainer.get_masked_adj()
            
            out = oracle(graph.x, masked_adj)
            
            y_new = torch.argmax(out[graph.new_idx])
            
            if y_new == graph.targets[graph.new_idx] and loss < best_loss:
                
                embedding_repr = torch.mean(oracle.get_embedding_repr(graph.x, masked_adj), dim=0)

                counterfactual = Data(x=graph.x, 
                                    adj=masked_adj, 
                                    y=torch.argmax(out, dim=1),
                                    sub_index=graph.new_idx,
                                    loss=loss,
                                    x_projection=embedding_repr)
            
            if time.time() - start_time > self.cfg.timeout:
                
                return counterfactual

        return counterfactual


                
class ExplainModelNodeMulti(torch.nn.Module):

    def __init__(self, graph, base_model):
        super(ExplainModelNodeMulti, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.graph = graph
        self.num_nodes = self.graph.x.shape[0]
        self.base_model = base_model
        self.adj_mask = self.construct_adj_mask()
        # For masking diagonal entries
        self.diag_mask = torch.ones(self.num_nodes, self.num_nodes) - torch.eye(self.num_nodes)
        self.diag_mask = self.diag_mask.to(device)
        

    def forward(self):
        masked_adj = self.get_masked_adj()
        S_f = self.base_model(self.graph.x, masked_adj)
        S_c = self.base_model(self.graph.x, self.graph.adj - masked_adj)
        return S_f, S_c

    def loss(self, S_f, S_c, pred_label, gam, lam, alp):
        
        relu = torch.nn.ReLU()
        
        _, sorted_indices = torch.sort(S_f, descending=True)
        S_f_y_k_s = sorted_indices[1]
        
        _, sorted_indices = torch.sort(S_c, descending=True)
        S_c_y_k_s = sorted_indices[1]
        
        
        L_f = relu(gam + S_f[S_f_y_k_s] - S_f[pred_label])
        L_c = relu(gam + S_c[pred_label] - S_c[S_c_y_k_s])
        
        masked_adj = self.get_masked_adj()
        L1 = torch.linalg.norm(masked_adj, ord=1)
        
        loss = L1 + lam * (alp * L_f + (1 - alp) * L_c)
        return loss

    def construct_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self):
        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.graph.adj
        masked_adj = adj * sym_mask
        
        return masked_adj


