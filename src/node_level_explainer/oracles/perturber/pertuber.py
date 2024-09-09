from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from abc import abstractmethod, ABC
from typing import Tuple
from src.node_level_explainer.utils.utils import normalize_adj

class Perturber(nn.Module, ABC):

    def __init__(self, cfg:DictConfig, model: nn.Module) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.deactivate_model()
        self.set_reproducibility()
        
    def deactivate_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_reproducibility(self)->None:
        torch.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed_all(self.cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    @abstractmethod
    def forward(self):

        pass

    @abstractmethod
    def forward_prediction(self):

        pass

class NodePerturberMultiple(Perturber):

    def __init__(self, 
                 cfg: DictConfig, 
                 num_classes: int, 
                 adj, 
                 model: nn.Module, 
                 nfeat: int):
        super().__init__(cfg=cfg, model=model)

        self.adj = normalize_adj(adj)
        self.nclass = num_classes
        self.num_nodes = self.adj.shape[0]
        self.features_add = False
        self.perturb_1 = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes, 14)))
        self.perturb_2 = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes, 15)))


    def forward(self, x, adj):

        if not self.features_add:
            x = torch.cat((F.sigmoid(self.perturb_1), self.perturb_2)) * x
        else:
            x = F.sigmoid(self.x_vec)

        return self.model(x, self.adj)

    
    def forward_prediction(self, x):

        self.perturb_1_thr = (F.sigmoid(self.perturb_1) >= 0.5).float()
        perturbation =  torch.cat((self.perturb_1_thr, self.perturb_2))

        if not self.features_add:
            x = perturbation * x
        else:
            x = self.features_t

        out = self.model(x, self.adj)
        return out, x, perturbation
    
    def loss(self, graph, output, y_node_non_differentiable):

        node_to_explain = graph.new_idx
        y_node_predicted = output[node_to_explain].unsqueeze(0)
        y_target = graph.targets[node_to_explain].unsqueeze(0)
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        loss_pred =  F.cross_entropy(y_node_predicted, y_target)
        loss_feat = F.l1_loss(graph.x, F.sigmoid(self.x_vec) * graph.x)
	
        loss_total =  constant * loss_pred + loss_feat

        results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
            "loss_graph_dist": 0.0,
            "loss_feat": loss_feat.item()
		}
        
        return loss_total, results, self.adj


class NodePerturber(Perturber):

    def __init__(self, 
                 cfg: DictConfig, 
                 num_classes: int, 
                 adj, 
                 model: nn.Module, 
                 nfeat: int,
                 nodes_number: int,
                 device: str = "cpu"):
        super().__init__(cfg=cfg, model=model)
        self.device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self.nclass = num_classes
        self.num_nodes = nodes_number
        self.features_add = False
        self.P_x = Parameter(torch.FloatTensor(torch.ones(self.num_nodes, nfeat))).to(device)
        self.adj = normalize_adj(adj).to(self.device)

          
    def gat_forward(self, V_x, edge_index, edge_attr):


        V_pert = F.sigmoid(self.P_x) * V_x

        return self.model(V_pert, edge_index, edge_attr)
    
    def forward(self, V_x, adj):

        if not self.features_add:
            V_pert = F.sigmoid(self.P_x) * V_x
        else:
            V_pert = F.sigmoid(self.P_x)

        return self.model(V_pert, self.adj)
    
    def gat_forward_prediction(self, V_x, edge_index, edge_attr)->Tuple[Tensor, Tensor, Tensor]:

        #pert = (F.sigmoid(self.P_x) >= 0.5).float()

        V_pert = F.sigmoid(self.P_x) * V_x

        out = self.model(V_pert, edge_index, edge_attr)
        return out, V_pert, self.P_x
    
    def forward_prediction(self, V_x)->Tuple[Tensor, Tensor, Tensor]:

        pert = (F.sigmoid(self.P_x) >= 0.5).float()

        if not self.features_add:
            V_pert = pert * V_x

        else:
            V_pert = pert 

        out = self.model(V_pert, self.adj)
        return out, V_pert, self.P_x
    
    
    def loss(self, graph, output, y_node_non_differentiable):

        node_to_explain = graph.new_idx
        y_node_predicted = output[node_to_explain].unsqueeze(0)
        y_target = graph.targets[node_to_explain].unsqueeze(0)
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        loss_pred =  F.cross_entropy(y_node_predicted, y_target) 
        loss_feat = F.l1_loss(graph.x, F.sigmoid(self.P_x) * graph.x)
	
        loss_total =  constant * loss_pred + loss_feat

        results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
            "loss_graph_dist": 0.0,
            "loss_feat": loss_feat.item()
		}
        
        return loss_total, results, self.adj
