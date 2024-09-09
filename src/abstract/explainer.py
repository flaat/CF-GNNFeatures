from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torch_geometric.data import Data
import torch
import numpy as np

class Explainer(ABC):

    def __init__(self, cfg: DictConfig) -> None:
        
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self.verbose = cfg.verbose
        self.num_classes = None


    @abstractmethod
    def explain_node(self, graph: Data, oracle)->dict:

        pass
    

    @abstractmethod
    def name(self):

        pass
    
    
    def set_reproducibility(self):

        # Reproducibility
        torch.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed_all(self.cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(self.cfg.general.seed)	