# Calorimetry Clustering at HGCAL/HCAL using SPVNAS / GNNs

## Directory structure
    .
    ├── configs -- configuration files for hydra
    ├── hgcal_dev
    │   ├── clustering
    │   ├── datasets -- pytorch lightning data modules
    │   ├── models -- pytorch lightning models
    │   ├── modules -- spvnas modules
    │   ├── training -- various helper functions for training
    │   ├── utils
    │   │   ├── common.py
    │   │   ├── comm.py
    │   │   ├── container.py
    │   │   ├── experiment.py
    │   │   └── metrics.py
    │   └── visualization
    │       ├── hcal.py
    │       └── hgcal.py
    ├── cluster.py -- script to cluster and eval embedded predictions
    ├── install.sh -- lists some needed packages that aren't up to date in conda
    ├── modules.sh -- for slurm
    ├── requirements.txt
    ├── setup.py
    ├── test.py -- script to save predictions from trained model
    ├── train.py -- script to train model

## Setup
### Requirements
* pytorch>=1.6
* pytorch-lightning
* pytorch-metric-learning
* wandb
* ninja

Setup an environment with the given requirements (a requirements.txt is provided, which may be useful). Then, run `pip install -e .` to install the hgcal-dev package. 

Next, modify the following hard-coded paths in the config files:
1. configs/config.yaml -- `outputs_dir` and `predictions_dir`
2. configs/dataset/* -- `data_dir`

The lightning data modules will download and extract the relevant datasets automatically the first time you run training. However, if you prefer to download them manually, they are available at:
* hcal -- https://cernbox.cern.ch/index.php/s/s19K02E9SAkxTeg/download
* hgcal -- https://cernbox.cern.ch/index.php/s/ocpNBUygDnMP3tx/download

If you download the datasets manually, please place them in the appropriate `data_dir` from above.

## Training
Configuration and command-line arguments are handled using [hydra](https://hydra.cc/docs/intro/). Logging is handled by [wandb](https://www.wandb.com/) (contact me and I can add you to the wandb team). Training is done using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/), with embedding losses from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/). For training, the most commonly-used arguments are `criterion`, `dataset`, `train`, and `wandb.name`. For example, to train spvcnn on hcal data for semantic segmentation using a single gpu, you can run `python train.py dataset=hcal criterion=cross_entropy_loss train=single_gpu wandb.name=hcal_semantic`. Model checkpoints will be saved in `outputs_dir`/wandb/{run}/files/{project}/{id}/checkpoints and uploaded to wandb.

## Evaluation
You can save predictions from a trained model by using `test.py`. Use the same configuration options as when training, and add `+wandb.id={your_id}`. The id is the last directory under Run Path on wandb (for example, if the run path is 'uw-hgcal/hgcal-spvcnn/2g1yke58', the id is 2g1yke58). You can then load these predictions to calculate metrics and so forth, or run `cluster.py` to cluster embedded predictions and predict instance labels.