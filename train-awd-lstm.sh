#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-2g-10gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="AWDLSTMTest"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load software/TensorFlow-A100-GPU
module load python/anaconda-python-3.7
# source activate awd-lstm

pip install -r awd_lstm_requirements.txt

python3 -u awd_lstm/main.py \
    --save "AWD_LSTMTest.pt" \
    --descriptive_name "AWDLSTM_Initial_Parameters" \
    --data data/test \
    --save_history "log_history.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 80 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --nonmono 8 \
    --cuda \

conda deactivate