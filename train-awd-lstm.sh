#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:a100-3g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="awdlstm"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL
CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/anaconda-python-3.7
module load software/TensorFlow-A100-GPU

start=$(date +%s)
echo "Starting script..."

echo "Batch size tuning"
python3 -u awd_lstm/main.py \
    --descriptive_name "awd_lstm_combined" \
    --data data/combined/isizulu \
    --save_history "logs/awd_lstm/$start.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 32 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --patience 4 \
    --nonmono 8 \
    --cuda \
    --when 25 35 \

python3 -u awd_lstm/main.py \
    --descriptive_name "awd_lstm_combined" \
    --data data/combined/isizulu \
    --save_history "logs/awd_lstm/$start.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 64 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --patience 4 \
    --nonmono 8 \
    --cuda \
    --when 25 35 \

python3 -u awd_lstm/main.py \
    --descriptive_name "awd_lstm_combined" \
    --data data/combined/isizulu \
    --save_history "logs/awd_lstm/$start.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 128 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --patience 4 \
    --nonmono 8 \
    --cuda \
    --when 25 35 \

python3 -u awd_lstm/main.py \
    --descriptive_name "awd_lstm_combined" \
    --data data/combined/isizulu \
    --save_history "logs/awd_lstm/$start.txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 30.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 256 \
    --bptt 70 \
    --dropout 0.4 \
    --dropouth 0.2 \
    --dropouti 0.65 \
    --dropoute 0.1 \
    --wdrop 0.5 \
    --seed 1882 \
    --patience 4 \
    --nonmono 8 \
    --cuda \
    --when 25 35 \

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"