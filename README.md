# slurm-tests

Create a conda environment with PyTorch installed:
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
conda create --prefix /fsx/myenv python=3.8
conda activate /fsx/myenv

# Modify this command if your instance has GPUs
conda install pytorch torchvision torchaudio cpuonly -c pytorch 
```

Launch command: `sbatch scripts/mnist_sbatch.sh`

Note: the train_mnist.py file is not set up to work as a distributed job yet.
