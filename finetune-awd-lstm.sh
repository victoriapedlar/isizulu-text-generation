#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=4 --gres=gpu:a100-4g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="awdlstm"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/anaconda-python-3.7
module load software/TensorFlow-A100-GPU

start=$(date +%s)
echo "Starting script..."

python3 -u awd_lstm/finetune.py \
    --descriptive_name "finetune_awd_lstm" \
    --data data/isolezwe/isizulu \
    --save_history "logs/finetune_awd_lstm/$(date "+%Y-%m-%d_%H-%M-%S").txt" \
    --emsize 800 \
    --nhid 1150 \
    --nlayers 3 \
    --lr 50.0 \
    --clip 0.25 \
    --epochs 750 \
    --batch_size 16 \
    --bptt 70 \
    --dropout 0.1 \
    --dropouth 0.1 \
    --dropouti 0.1 \
    --dropoute 0.05 \
    --wdrop 0.1 \
    --seed 1882 \
    --patience 3 \
    --nonmono 8 \
    --cuda \
    --when 40 80 120 160 \
    --load "models/awd_lstm/_emsize_800_nhid_1150_nlayers_3_lr_30.0_wdc_1.2e-06_clip_0.25_epochs_500_bsz_32_bptt_70_dropout_0.2_dropouth_0.2_dropouti_0.2_dropoute_0.1_wdrop_0.2_seed_1882_patience_3_when_[40, 80, 120].pt" \
    --save "models/awd_lstm/finetuned_model.pt" \


end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"