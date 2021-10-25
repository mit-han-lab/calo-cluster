# Calorimetry Clustering at HGCAL/HCAL using SPVNAS / GNNs

## Directory structure
    .
    ├── configs -- configuration files for hydra
    ├── calo_cluster
    │   ├── clustering
    │   ├── datasets -- pytorch lightning data modules
    │   ├── evaluation -- visualization, evaluation, performance studies, etc.
    │   ├── models -- pytorch lightning models
    │   ├── training -- various helper functions for training
    ├── setup.py
    ├── test.py -- script to save predictions from trained model (can be done from a notebook, as well)
    ├── train.py -- script to train model

## Installation
### Prerequisites
* CUDA 10.2.
* [torchsparse v1.4.0](https://github.com/mit-han-lab/torchsparse) (instructions to install this are below, but you may need to first install the Google Sparse Hash library as described in the torchsparse github)

First, clone this repository: `git clone --recurse-submodules https://github.com/mit-han-lab/calo-cluster.git `.

Then, run the following commands to create a conda environment with the necessary packages:
```
conda create -n calo_cluster python=3.8
conda activate calo_cluster
conda install numpy pandas plotly jupyter tqdm yaml matplotlib seaborn scikit-learn -y
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install pytorch-lightning submitit uproot -c conda-forge -y
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip install hydra-core==1.1.0 wandb

pip install -e .

```

If such an error occurs, 

```
  File "/opt/conda/envs/calo_cluster_test/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1147, in _get_data
    raise RuntimeError('Pin memory thread exited unexpectedly')
RuntimeError: Pin memory thread exited unexpectedly
```
modify the parameter `pin_memory=True`, to `pin_memory=False` in `/calo_cluster/datasets/base.py` . 

If such an error occurs,
```
RuntimeError: Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
```

use `ulimit -n 10000`. 

By default, all data should be in `/data` (check the dataset configs & modify if necessary -- you can run `sudo mkdir -m777 /data` to make this directory accessible to all users). You similarly may wish to modify the paths in configs/config.yaml.

The datasets are available at:
* vertex -- https://cernbox.cern.ch/index.php/s/WKySWaNStTH3y49/download
* simple (toy dataset) -- https://cernbox.cern.ch/index.php/s/f3G7FPipgm5f5Ai/download

## Training
Configuration and command-line arguments are handled using [hydra](https://hydra.cc/docs/intro/). Logging is handled by [wandb](https://www.wandb.com/) (contact me and I can add you to the wandb team). Training is done using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/). For training, the most commonly-used arguments are `embed_criterion`, `semantic_criterion`, `dataset`, `train`, `model`, and `wandb.name`. See the configs for a better understanding. For example, to train spvcnn on the toy data for instance segmentation using a single gpu, you can run `python train.py dataset=simple ~semantic_criterion train=single_gpu wandb.name="simple_instance_test" model.cr=0.5 train.num_epochs=5`. Model checkpoints will be saved in `{outputs_dir}/{project}/{id}/checkpoints`.

## Evaluation
See `notebooks/simple/simple_instance` for an example of how you can evaluate performance once you have trained a model.
