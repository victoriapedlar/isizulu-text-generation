#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-4g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="AWDLSTMTest"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

git clone https://github.com/victoriapedlar/isizulu-text-generation.git

cd isizulu-text-generation

module load python/anaconda-python-3.7
source activate awd-lstm

./utils/getdata.sh

python3 -u awd_lstm/main.py \
    --save "AWD_LSTM_Test.pt" \
    --descriptive_name "AWDLSTM_Luc_Hayward" \
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