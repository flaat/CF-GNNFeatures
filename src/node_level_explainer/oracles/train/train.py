import numpy as np
from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from ..models.gcn import GCNSynthetic, GCN
from ...utils.utils import normalize_adj
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Dataset
from torcheval.metrics import MulticlassAccuracy
from torch import nn
import wandb

class Trainer:

	def __init__(self, cfg: DictConfig, dataset: Dataset, model: nn.Module, loss: nn.Module) -> None:

		# Reproducibility
		torch.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed(cfg.general.seed)
		torch.cuda.manual_seed_all(cfg.general.seed) 
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.autograd.set_detect_anomaly(True)
		np.random.seed(cfg.general.seed)
		
		self.device = "cuda" if torch.cuda.is_available() and cfg.device=="cuda" else "cpu"
		self.cfg = cfg
		self.dataset = dataset
		self.num_classes = self.dataset.y.unique().shape[0] if cfg.dataset.name != "Facebook" else 193
		self.metric = MulticlassAccuracy(num_classes=self.num_classes)
		dense_matrix = to_dense_adj(self.dataset.edge_index).squeeze()
		self.norm_adj = normalize_adj(dense_matrix)
		self.model = model
		self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.lr)
		self.loss = loss

	def _train(self, epoch: int):
     
		self.metric.reset()
		self.optimizer.zero_grad()
		output = self.model(self.dataset.x.to(self.device), self.norm_adj.to(self.device))
		loss_train = self.loss(output[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
		y_pred = torch.argmax(output, dim=1) 
		self.metric.update(y_pred[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
		loss_train.backward()
		clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip)
		self.optimizer.step()
		return loss_train, self.metric.compute()
		

	def _test(self):

		with torch.no_grad():
			
			self.metric.reset()
			output = self.model(self.dataset.x.to(self.device), self.norm_adj.to(self.device))
			loss_test = self.loss(output[self.dataset.test_mask], self.dataset.y[self.dataset.test_mask])
			y_pred = torch.argmax(output, dim=1) 
			self.metric.update(y_pred[self.dataset.test_mask], self.dataset.y[self.dataset.test_mask])
			return loss_test, self.metric.compute()
	
	def start_training(self):
		import os
		import pandas as pd
		metrics = pd.DataFrame(columns=["Epoch", "TrainLoss", "TrainAcc", "TestLoss", "TestAcc"])

		for epoch in range(self.cfg.trainer.epochs):

			train_loss, train_accuracy = self._train(epoch=epoch)
			test_loss, test_accuracy = self._test()
			print(f"Epoch: {epoch:4d} Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_accuracy:.4f}")
			metrics.loc[epoch] = [epoch, train_loss.detach().cpu().item(), train_accuracy.item(), test_loss.detach().cpu().item(), test_accuracy.item()]

		if not os.path.exists(f"data/models"):
			os.makedirs(f"data/models")


		torch.save(self.model.state_dict(), f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.n_layers}_epochs_{self.cfg.trainer.epochs}.pt")
		metrics.to_csv(f"data/models/{self.cfg.dataset.name}_{self.cfg.model.name}_{self.cfg.model.n_layers}_epochs_{self.cfg.trainer.epochs}_METRICS.csv")