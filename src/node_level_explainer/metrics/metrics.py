import torch
from torch_geometric.data import Data
from scipy.spatial import distance

def sample_distance_from_mean(mean: torch.Tensor, sample: Data)-> float:

    distance = torch.sqrt(torch.sum((torch.mean(sample.x, dim=0) - mean)**2))
    return distance.item() 

def sample_distance_from_mean_projection(mean: torch.Tensor, sample: Data)-> float:

    distance = torch.sqrt(torch.sum((sample.x_projection - mean)**2))
    return distance.item() 

def factual_counterfactual_distance(factual: Data, counterfactual: Data)->float:

    distance = torch.sqrt(torch.sum((factual.x_projection - counterfactual.x_projection)**2))
    return distance.item() 

def accuracy(factual: Data, counterfactual: Data) -> float:

    return None

def validity(factual: Data, counterfactual: Data)-> float:

    return None

def fidelity(factual: Data, counterfactual: Data):

    #TODO: Check Fidelity
    factual_index = factual.new_idx
    cfactual_index = counterfactual.sub_index
    phi_G = factual.y[factual_index]
    y = factual.y_ground[factual_index]
    phi_G_i = counterfactual.y[cfactual_index]
    
    prediction_fidelity = 1 if phi_G == y else 0
    
    counterfactual_fidelity = 1 if phi_G_i == y else 0
    
    result = prediction_fidelity - counterfactual_fidelity
    
    return result

def explanation_size():
    pass
    #TODO: implement explanation size

def edge_sparsity(factual: Data, counterfactual: Data):

    modified_edges = (torch.sum((factual.adj != counterfactual.adj))//2)
    return ((modified_edges) / (factual.adj.numel()//2)).item()

def node_sparsity(factual: Data, counterfactual: Data):

    modified_attributes =  torch.sum(factual.x  != counterfactual.x)
    return ((modified_attributes) / (factual.x.numel())).item()

def graph_edit_distance(factual: Data, counterfactual: Data):

    triu_indices = torch.triu_indices(factual.adj.size(0), factual.adj.size(1), offset=1)
    return torch.sum(factual.adj[triu_indices[0], triu_indices[1]] != counterfactual.adj[triu_indices[0], triu_indices[1]]).item()

def perturbation_distance(factual: Data, counterfactual: Data):

    return torch.mean((factual.x.long() ^ counterfactual.x.long()).sum(dim=1).float()).item()

def mahalanobis_distance(data_info, counterfactual: Data):

    return distance.mahalanobis(counterfactual.x_projection.detach().cpu(), data_info.distribution_mean_projection.detach().cpu(), data_info.inv_covariance_matrix.detach().cpu())
