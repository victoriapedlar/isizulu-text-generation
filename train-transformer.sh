#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-4g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="Transformer_Vanilla"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

git clone https://github.com/victoriapedlar/isizulu-text-generation.git

cd isizulu-text-generation

module load python/anaconda-python-3.7
source activate transformer

./utils/getdata.sh

export PYTHONPATH=$PYTHONPATH:`pwd`/scripts

python3 scripts/train_example.py

python3 scripts/create_csv.py