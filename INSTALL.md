# Installation

## Requirements

We use an evironment with the following specifications, packages and dependencies:

- Ubuntu 20.04.3 LTS
- Python 3.8
- conda 4.12.0
- [PyTorch v1.8.1](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.9.1](https://pytorch.org/get-started/previous-versions/)
- [PyTorch Lightning v1.5.0](https://pytorch-lightning.readthedocs.io/en/stable/)

## Setup Instructions

- Create a conda environment
  
  ```bash
  conda create --name tasformer python=3.8 -y
  conda activate tasformer
  ```

- Install packages and other dependencies.

  ```bash
  git clone https://github.com/subake/TASFormer.git
  cd TASFormer

  # Install Pytorch
  pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

  # Install other dependencies
  pip install -r requirements.txt
  ```

- Setup wandb and login into your account.

  ```bash
  wandb login
  ```