# Calorimetry Clustering at HGCAL/HCAL using SPVNAS / GNNs

## Directory structure
    .
    ├── configs -- configuration files for hydra
    ├── hgcal_dev
    │   ├── clustering
    │   ├── datasets -- pytorch lightning data modules
    │   ├── evaluation -- visualization, evaluation, performance studies, etc.
    │   ├── models -- pytorch lightning models
    │   ├── modules -- spvnas modules
    │   ├── training -- various helper functions for training
    ├── environment.yml -- conda environment file
    ├── setup.py
    ├── test.py -- script to save predictions from trained model (can be done from a jupyter notebook, as well)
    ├── train.py -- script to train model

## Installation
### Prerequisites
* [torchsparse](https://github.com/mit-han-lab/torchsparse) (make sure to install it in an environment with CUDA)

First, clone this repository: `git clone --recurse-submodules https://github.com/mit-han-lab/hgcal-dev.git `.

A conda `environment.yml` file is provided; use it to create an environment by running `conda env create -f environment.yml`. Then, run `pip install -e .` to install the hgcal-dev package. 

Next, modify the following hard-coded paths in the config files:
1. configs/config.yaml -- `outputs_dir` and `predictions_dir`
2. configs/dataset/* -- `data_dir`

The lightning data modules will download and extract the relevant datasets automatically the first time you run training. However, if you prefer to download them manually, they are available at:
* hcal -- https://cernbox.cern.ch/index.php/s/s19K02E9SAkxTeg/download
* hgcal -- https://cernbox.cern.ch/index.php/s/ocpNBUygDnMP3tx/download
* simple (toy dataset) -- https://cernbox.cern.ch/index.php/s/4mMbz29aSqg1EeX/download

If you download the datasets manually, please place them in the appropriate `data_dir` from above.

## Training
Configuration and command-line arguments are handled using [hydra](https://hydra.cc/docs/intro/). Logging is handled by [wandb](https://www.wandb.com/) (contact me and I can add you to the wandb team). Training is done using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/). For training, the most commonly-used arguments are `criterion`, `dataset`, `train`, and `wandb.name`. For example, to train spvcnn on the toy data for instance segmentation using a single gpu, you can run `python train.py dataset=simple criterion=centroid train=single_gpu wandb.name="simple_test"`. Model checkpoints will be saved in `outputs_dir/{project}/{id}/checkpoints`.

## Evaluation
See `notebooks/simple/simple_instance` for an example of how you can evaluate performance once you have trained a model.
