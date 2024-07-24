# Binary Neural Networks

This is the code repository for additional sets of experiments on binary neural networks
with alternative to backpropagation.

Previous smaller scale experiments are located at https://github.com/BenCrulis/binary_nn.

This repository contains the code used for the experiments of the article,
the raw experiment data and the hyperparameters found by grid search.

## Structure of the repository
- `binary_nn` : python sources
- `configs` : yaml files with the hyperparameters found using grid search
  (the .csv file extension is wrong)
- `experiment_results`` : csv files with the raw results from the grid search
  as downloaded from Weights & Biases. The file `test_results.csv` contains
  the data reported in the extension article. 
- `notebooks` : folder containing the notebooks for the data analysis and creation of tables
- `scripts` : utility scripts to download data from W&B, execute experiments from config
  files and compute config files from grid search results.
- `config.yaml` : contains information about models to perform their binarization
- `ds_config.yaml` : contains the paths to the datasets (change this file to run experiments)
- `train.py` : main script to train models

## Getting started

Run `pip install -r requirements.txt` to install the dependencies
(use of a virtual environment is advised).

Run `python train.py --help` to see the available option from training.

Example command, perform the training of a binary neural network model
with the MobileNetV2 architecture with the Direct Feedback Alignment (DFA) training method:
`python train.py --binary-act --binary-weights --lr=0.001 --method=dfa --model=MobileNetV2 --wd 1e-5 --augment --epochs 80 --bs 64`

To disable W&B logging, set the environment variable `WANDB=0` or use
the `--no-wandb` command flag.

## Backpropagation alternatives

We implement several alternatives to the backpropagation algorithm:
- Direct Feedback Alignment (DFA), `--method=dfa`
- Direct Random Target Propagation (DRTP), `--method=drtp`
- Information Bottleneck training (HSIC), `--method=hsic`
- Sigprop Target Loop variant, also called PEPITA, `--method=pepita`

Vanilla backpropagation can be used by setting `--method=bp`.