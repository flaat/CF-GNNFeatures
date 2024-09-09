import random
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import pickle
from torch_geometric.utils import dense_to_sparse


def get_dataset(dataset_name: str = None)->Data:
    """_summary_

    Args:
        dataset_name (str, optional): _description_. Defaults to None.

    Returns:
        Data: _description_
    """
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root="data", name=dataset_name) [0]       
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()

        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index)
    
    elif dataset_name == "karate":
        from torch_geometric.datasets import KarateClub
        # It WORKS
        dataset = KarateClub()[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))

        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index)
    
    # TODO: Test twich
    elif dataset_name == "twitch":
        from torch_geometric.datasets import Twitch

        dataset = Twitch(root="data", name="EN")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index)
    
    elif dataset_name == "actor":
        from torch_geometric.datasets import Actor
        #It WORKS!

        dataset = Actor(root="data")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.03, random_state=random.randint(0, 100))
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index)
    
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        from torch_geometric.datasets import WebKB
        # It WORKS!

        dataset = WebKB(root="data", name=dataset_name)[0]  
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index)
    
    
    elif dataset_name in ["Wiki", "BlogCatalog", "Facebook", "PPI"]:
        from torch_geometric.datasets import AttributedGraphDataset

        dataset = AttributedGraphDataset(root="data", name=dataset_name)[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1)
        y = dataset.y if dataset_name != "Facebook" else torch.argmax(dataset.y, dim=1)
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=y, train_mask=train_index, test_mask=test_index)        
        
    elif "syn" in dataset_name:
        with open(f"data/{dataset_name}.pickle","rb") as f:
            data = pickle.load(f)

        adj = torch.Tensor(data["adj"]).squeeze()  
        features = torch.Tensor(data["feat"]).squeeze()
        labels = torch.tensor(data["labels"]).squeeze()
        idx_train = data["train_idx"]
        idx_test = data["test_idx"]
        edge_index = dense_to_sparse(adj)   

        train_index, test_index = train_test_split(idx_train + idx_test, test_size=0.2, random_state=random.randint(0, 100))  
        
        return Data(x=features, edge_index=edge_index[0], y=labels, train_mask=idx_train, test_mask=idx_test)
    
    else:
        raise Exception("Choose a valid dataset!")
    