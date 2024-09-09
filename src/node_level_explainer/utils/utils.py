import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph
import sys
from texttable import Texttable
import torch.optim as optim


class TimeOutException(Exception):
    
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
    
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def get_optimizer(cfg, cf_model):

        if cfg.optimizer.name == "sgd" and cfg.optimizer.n_momentum == 0.0:
            return optim.SGD(cf_model.parameters(), lr=cfg.optimizer.lr)

        elif cfg.optimizer.name == "sgd" and cfg.optimizer.n_momentum != 0.0:
            return optim.SGD(cf_model.parameters(), lr=cfg.optimizer.lr, nesterov=True, momentum=cfg.optimizer.n_momentum)
        
        elif cfg.optimizer.name == "adadelta":
            return optim.Adadelta(cf_model.parameters(), lr=cfg.optimizer.lr)
        
        elif cfg.optimizer.name == "adam":
            return optim.Adam(cf_model.parameters(), lr=cfg.optimizer.lr)
        
        else:
            raise ValueError(f"Optimizer {cfg.optimizer.name} does not exist!")


def get_degree_matrix(adj: torch.Tensor):
    return torch.diag(adj.sum(dim=1))

def print_info(dictionary):

    # Initialize the table
    table = Texttable(max_width=0)
    align_type = ["c"]
    data_type = ["t"]
    cols_valign = ["m"]
    
    table.set_cols_align(align_type*len(dictionary))
    table.set_cols_dtype(data_type*len(dictionary))
    table.set_cols_valign(cols_valign*len(dictionary))
    # Add rows: one row for headers, one for values
    table.add_rows([dictionary.keys(),
                    dictionary.values()])
    
    sys.stdout.write("\033[H\033[J")  # Move cursor to the top and clear screen
    sys.stdout.write(table.draw() + "\n")
    sys.stdout.flush()


def normalize_adj(adj):
    """
    Normalize adjacency matrix using the reparameterization trick from the GCN paper.
    """
    # Add self-loops to the adjacency matrix
    A_tilde = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute the degree matrix and its inverse square root
    D_tilde = torch.pow(get_degree_matrix(A_tilde), -0.5)
    D_tilde[torch.isinf(D_tilde)] = 0  # Set inf values to 0

    # Compute the normalized adjacency matrix
    norm_adj = D_tilde @ A_tilde @ D_tilde

    return norm_adj



def get_neighbourhood(node_idx: int, edge_index, n_hops, features, labels):
    """
    Get the subgraph induced by a node "node_idx" along with all the features
    """
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)     # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, relabel_nodes=True)       # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze() 
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict



def create_symm_matrix_from_vec(vector, n_rows, device: str = "cpu"):
    matrix = torch.zeros(n_rows, n_rows, device=device)
    idx = torch.tril_indices(n_rows, n_rows, device=device)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)
    