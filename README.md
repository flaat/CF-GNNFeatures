# Combining Structural and Node Feature Perturbations to Generate Counterfactual Explanations for Graph Neural Networks
![Python Version](https://img.shields.io/badge/python-3.11.9-brightgreen)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.3.0-brightgreen)
![PyTorch Geometric Version](https://img.shields.io/badge/torch_geometric-2.5.3-brightgreen)
![Hydra](https://img.shields.io/badge/hydracore-1.3.2-brightgreen)
![PyTorch Geometric Version](https://img.shields.io/badge/wandb-0.17.1-brightgreen)

This repository is the official implementation of the paper:
### Combining Structural and Node Feature Perturbations to Generate Counterfactual Explanations for Graph Neural Networks

## Requirements

To install requirements:

```setup
pip install requirements.txt
```

## How to start the code

You can start the the code just typing 

```start
python main.py
```

In this way you are going to start the software with the default hydra configuration.

If you want to change options in the default configuration continue to read

## Conifgurations
There are many ways to configure the software, you can chose among: dataset, logger, technique, run_mode

### Dataset
You can chose among one of these dataset classes, namely:
* **planetoid**: cora, citeseer, pubmed
* **attributed**: Facebook BlogCatalog, Wiki
* **webkb**: Cornell, Wisconsin, Texas
* **karate**
* **actor**

for example if you need to use the dataset cora you can use the following commands:
```dataset
... dataset=planetoid dataset.name=cora
```

### Logger
You can chose among different mode for the logger
* **mode**: online, offline
* **conifg**: $your_config_name

### Technique

### Run Mode

## How to run the code, an example

In order to run the code you can 

```start
python main.py run_mode=run logger.mode=online  
```

## How to run the experiments

In order to run the code type

```start
bash experiments.sh
```
