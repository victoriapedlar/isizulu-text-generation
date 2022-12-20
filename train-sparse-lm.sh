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

cd sparse_text_generation/language_modeling/
python3 examples/run_lm_finetuning.py \
        --train_data_file=/isizulu-text-generation/data/test/train.txt \
        --eval_data_file=/isizulu-text-generation/data/test/valid.txt \
        --output_dir=/isizulu-text-generation/sparse_text_generation \
        --model_type=gpt2 \
        --model_name_or_path=gpt2-medium \
        --block_size=512 \
        --do_train \
        --evaluate_during_training \
        --loss=entmax \
        --entmax_alpha=1.2 \
        --top_k=0 \
        --top_p=0

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"