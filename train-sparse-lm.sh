#!/bin/sh

#SBATCH --account=nlpgroup
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=6 --gres=gpu:a100-3g-20gb:1
#SBATCH --time=48:00:00
#SBATCH --job-name="sparse_lm"
#SBATCH --mail-user=PDLVIC001@myuct.ac.za
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/anaconda-python-3.7
module load software/TensorFlow-A100-GPU

export LD_LIBRARY_PATH=/home/pdlvic001/.local/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

start=$(date +%s)
echo "Starting script..."

python3 -m sparse_text_generation.language_modeling.examples.run_lm_finetuning \
        --train_data_file data/combined/isizulu/train.txt \
        --eval_data_file data/combined/isizulu/valid.txt \
        --output_dir models/sparse_lm/experiment-2 \
        --tokenizer_name tokenizers/ByteLevelBPETokenizer \
        --model_type gpt2 \
        --model_name_or_path gpt2-medium \
        --mode from_scratch \
        --block_size 128 \
        --do_train \
        --num_train_epochs 10 \
        --learning_rate 0.0001 \
        --weight_decay 0.1 \
        --evaluate_during_training \
        --loss entmax \
        --entmax_alpha 1.5 \
        --top_k 0 \
        --top_p 0

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"