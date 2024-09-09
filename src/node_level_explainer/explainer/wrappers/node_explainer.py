from functools import partial
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from src.abstract.explainer import Explainer
from src.utils.explainer import get_node_explainer
from ....abstract.wrapper import Wrapper
from torch_geometric.utils import to_dense_adj
from ...utils.utils import TimeOutException, get_neighbourhood, normalize_adj
from ...evaluation.evaluate import compute_metrics
from src.datasets.dataset import DataInfo
from torch.nn import Module
import torch.multiprocessing as mp
import wandb
import concurrent.futures
import copy
    

class NodesExplainerWrapper(Wrapper):
    
    queue = None
    results_queue = None
    
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)

        torch.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed_all(cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(cfg.general.seed)

    def check_graphs(self, sub_adj) -> bool:
        """
        Check if the input adjacency matrix represents an empty or trivial graph.

        A graph is considered empty if there are no edges between any nodes. A graph
        is trivial if it consists of a single node without any self-loops. This function
        checks for these conditions by inspecting the shape and the number of elements
        in the adjacency matrix.

        Parameters:
        - sub_adj (Tensor): The adjacency matrix of a graph.

        Returns:
        - bool: True if the graph is empty or trivial, False otherwise.
        """
        return len(sub_adj.shape) == 1 or sub_adj.numel() <= 1


    def explain(self, data: Dataset, datainfo: DataInfo, explainer: str, oracle: Module)->dict:
        """
        Explain the node predictions of a graph dataset.

        This method applies the provided explainer to each test instance in the dataset to generate
        counterfactual explanations. It computes and saves the explanation metrics.

        Parameters:
        - data (Dataset): The graph dataset containing features, edges, and test masks.
        - datainfo (DataInfo): Object containing dataset metadata and other relevant information.
        - explainer (Explainer): The explainer algorithm to generate explanations.

        Returns:
        - dict: A dictionary containing the results of the explanation process and metrics.
        """
        print(f"{explainer=}")
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        adj = to_dense_adj(data.edge_index).squeeze()    
        norm_adj = normalize_adj(adj).to(device)   

        x = data.x.to(device)
        adj = norm_adj.to(device)

        output = oracle(x, adj).detach()
        y_pred_orig = torch.argmax(output, dim=1)
        targets = (1 + y_pred_orig)%datainfo.num_classes
        metric_list = []

        embedding_repr = oracle.get_embedding_repr(x, adj).detach()
        datainfo.distribution_mean_projection = torch.mean(embedding_repr, dim=0).cpu()
        covariance =  np.cov(embedding_repr.detach().cpu().numpy(), rowvar=False)
        datainfo.inv_covariance_matrix = torch.from_numpy(np.linalg.inv(covariance)).float().cpu()

        mp.set_start_method('spawn', force=True)
        oracle.share_memory()
        
        try:
        
            with mp.Manager() as manager:
                
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self.worker_process, queue, results_queue)

                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                    
                
                pid = 0
                
                for mask_index in tqdm(data.test_mask):
                    pid+=1

                    sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(node_idx=int(mask_index), 
                                                                                    edge_index=data.edge_index.cpu(),
                                                                                    n_hops=self.cfg.model.n_layers+1, 
                                                                                    features=data.x.cpu(), 
                                                                                    labels=data.y.cpu())
                            
                    if self.check_graphs(sub_adj=sub_adj):
                        continue

                    new_idx = node_dict[int(mask_index)]
                    sub_index = list(node_dict.keys())
                    
                    # Pass everything on cpu because of Queue
                    args = (sub_index, 
                            new_idx, 
                            y_pred_orig.cpu(), 
                            targets.cpu(), 
                            device, 
                            oracle.cpu(), 
                            sub_labels.cpu(), 
                            sub_feat.cpu(), 
                            explainer, 
                            datainfo, 
                            node_dict, 
                            self.cfg, 
                            sub_adj.cpu(), 
                            pid)
                    
                    queue.put(args)
                
                # Signal the end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                    
                for worker in workers:
                    worker.join()

                # Collect the results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                dataframe = pd.DataFrame.from_dict(metric_list)
                        
                wandb.log(dataframe.mean().to_dict())
            
        except Exception as e:
            
            print(f"{e}")


    @staticmethod
    def worker_process(queue, results_queue):
        while True:
            args = queue.get()
            if args is None:
                break
            result = process(*args)
            results_queue.put(result)

def process(sub_index, new_idx, y_pred_orig, targets, device, oracle, sub_labels, sub_feat, explainer_name, datainfo, node_dict, cfg, sub_adj, pid):
    
    import copy 
    model = copy.deepcopy(oracle).to(device)
    explainer = get_node_explainer(explainer_name)
    explainer = explainer(cfg)
    explainer.num_classes = datainfo.num_classes
    sub_y = y_pred_orig[sub_index]
    sub_targets = targets[sub_index]
    sub_feat = sub_feat.to(device)
    sub_adj = sub_adj.to(device)
    repr = model.get_embedding_repr(sub_feat, sub_adj)
    embedding_repr = torch.mean(repr, dim=0).to(device) 
    factual = Data(x=sub_feat, adj=sub_adj, y=sub_y, y_ground=sub_labels, new_idx=new_idx, targets=sub_targets, node_dict=node_dict, x_projection=embedding_repr)  
    factual = factual.to(device)
    counterfactual = explainer.explain_node(graph=factual, oracle=model.to(device))
    metrics = compute_metrics(factual, counterfactual, device=device, data_info=datainfo)
    print(f"Terminated {pid=}")
    return metrics