import multiprocessing
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from src.datasets.dataset import DataInfo
from src.node_level_explainer.oracles.models.gcn import GAT, GCNSynthetic
from src.utils.dataset import get_dataset
import random
from src.utils.utils import flatten_dict, merge_hydra_wandb, read_yaml


def log_params(cfg: DictConfig) -> None:
    
    import pandas as pd
    
        
    temp_config = OmegaConf.to_container(cfg)
    config_to_log = flatten_dict(d=temp_config)
    config_to_log = pd.DataFrame([config_to_log])
    config_to_log.astype(str)
    param_table = wandb.Table(dataframe=config_to_log)

    wandb.log({"params": param_table})
    
def set_run_name(cfg, run):
    
    from datetime import datetime

    run_name: str = f"{cfg.technique.technique_name}_{cfg.dataset.name}_{cfg.optimizer.name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run.name = run_name
    run.save()


def run_sweep_agent(cfg: DictConfig, sweep_id: str):
    wandb.agent(sweep_id=sweep_id, function=lambda: train(cfg))


def train(cfg):
        
    with wandb.init(project=cfg.logger.project, group="experiment_1", mode=cfg.logger.mode) as run:

        from src.node_level_explainer.explainer.wrappers.node_explainer import NodesExplainerWrapper
        from src.node_level_explainer.oracles.train.train import Trainer
        from torch.nn import functional as F
        random.seed(cfg.general.seed)        
        
        merge_hydra_wandb(cfg, wandb.config)
        log_params(cfg)
        set_run_name(cfg, run)

        device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        dataset = get_dataset(cfg.dataset.name, test_size=0.2)
        datainfo = DataInfo(cfg, dataset)

        wrapper = NodesExplainerWrapper(cfg=cfg)
        oracle = GCNSynthetic(cfg, nfeat=datainfo.num_features, nclass=datainfo.num_classes)
        
        datainfo = DataInfo(cfg, dataset)
        dataset = dataset.to(device)
        oracle = oracle.to(device)
        datainfo.kfold = cfg.general.seed
        trainer = Trainer(cfg=cfg, dataset=dataset, model=oracle, loss=F.cross_entropy)
        trainer.start_training()
        oracle = trainer.model
        oracle.eval()
        wrapper.explain(data=dataset, datainfo=datainfo, explainer=cfg.technique.technique_name, oracle=oracle)       

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.run_mode == 'sweep':

        sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.logger.config}.yaml')
        sweep_config["name"] = f"{cfg.dataset.name}"
        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.logger.project)
        
        multiprocessing.set_start_method('spawn', force=True)

        # Number of parallel agents
        num_agents = cfg.num_agents if 'num_agents' in cfg else 1 # Default to 4 agents if not specified
        processes = []
        for _ in range(num_agents):
            p = multiprocessing.Process(target=run_sweep_agent, args=(cfg, sweep_id))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
                    
    elif cfg.run_mode == "run":
        
        train(cfg)
        
    else:
        
        raise ValueError(f"Values for run_mode can be sweep or run, you insert {cfg.run_mode}")
    
if __name__ == "__main__":

    main()
