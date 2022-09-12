#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-2g-10gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="Transformer_Vanilla"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load software/TensorFlow-A100-GPU
module load python/anaconda-python-3.7
source activate transformer

export PYTHONPATH=$PYTHONPATH:`pwd`/scripts

python3 scripts/train_example.py

python3 scripts/create_csv.py