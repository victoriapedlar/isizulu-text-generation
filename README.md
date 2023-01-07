# isizulu-text-generation

# AWD-LSTM

+ This code was originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model) and is heavily inspired by the original AWD-LSTM implementation [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm)

+ The code in this notebook is available on [google colab](https://colab.research.google.com/drive/1yyUGJfyYKdvPi6J7ZlsxPg9E_ppZG1xU) and on [github](https://github.com/mikkelbrusen/awd-inspired-lstm).

This version of AWD-LSTM was created by [Gustav Madslund](https://github.com/gustavmadslund) and [Mikkel Møller Brusen](https://github.com/mikkelbrusen).


### Core components
1.   **[x]  - Multi Layer** - We will need to controll what happens in between the layers, therefore, instead of using the multi layer cuDNN lstm implementation, we will create multiple single layer cuDNN lstms.
2.   **[x] - Weight drop** using DropConnect on hidden-hidden weights $[U^i, U^f, U^o, U^c]$ before forward and backward pass - makes it possible to use cuDNN LSTM
3.   **[x] - Optimization** using SGD and ASGD while training

### Extended regularization techniques
4.   **[ ] - Variable sequence length** to allow all elements in the dataset to experience a full BPTT window
  - **[ ] - Rescale learning rate** to counter the varible sequence lengths favoring short sequences with fixed learning rate
5.   **[x] - Variational dropout AKA LockDrop** for everything else than hidden-hidden, such that we use same dropout mask for all input/output in a forward backward pass of LSTM
6.   **[x] - Embedding dropout** which is **not** just a dropout applied on the embedding
7.   **[x]  - Weight tying** to reduce parameters and prevent model from having to learn one-to-one correspondance between input and output
8.   **[x] - Embed size** independent from hidden size, to reduce parameters.
9.   **[ ] - AR and TAR** - $L_2$-regularization by applying AR and TAR loss on the final RNN layer - can screw stuff up

### Getting started
Ensure that all scripts are run from the root directory.

Install requirements:
`pip3 install -r awd_lstm_requirements.txt`

Fetch training data:
`./utils/getdata.sh`

Minimum arguments to run the model:
```
python3 awd_lstm/main.py \
   --data data/nchlt/isizulu/ \
   --save "/content/drive/My Drive/Colab Notebooks/AWD_LSTM_Test.pt" \ 
   --descriptive_name "ExampleAWDLSTM" \
   --save_history "/content/drive/My Drive/Colab Notebooks/log_history.txt"
```

### Experiments
The following provide the needed parameters to recreate the top performing model on the isiZulu dataset. To run on alternate datasets the --data argument should be changed. Each of the models takes at least 3-4 hours to reach adequate performance and up to 10-12 to reach the best performance. Models were trained using a mix of Nvidia K80, P100 and V100 GPUs.

```
python3 -u awd_lstm/main.py \
    --save "AWD_LSTM_Test.pt" \
    --descriptive_name "ExampleAWDLSTM" \
    --data data/nchlt/isizulu/ \
    --save_history "/content/drive/My Drive/Colab Notebooks/log_history.txt" \
    --model "LSTM" \
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
```

```
python -u awd_lstm/finetune.py \
    --batch_size 20 \
    --data data/penn \
    --dropouti 0.4 \
    --dropouth 0.25 \
    --seed 141 \
    --epoch 500 \
    --save PTB.pt
```
Note that `finetune.py` overwrites the model it loads. If you wish to keep the original model, copy it elsewhere before starting the finetuning.

# Transformers

An implementation of a multilingual GPT-2 and utilities for downloading and preprocessing training data.

### Getting started
Ensure that all scripts are run from the root directory.

Install requirements:
`pip3 install -r transformer_requirements.txt`

Fetch training data:
`./utils/getdata.sh`

Add scripts directory to PYTHONPATH:
`export PYTHONPATH=$PYTHONPATH:`pwd`/scripts`

Train GPT-2 model:
`python3 scripts/train_example.py`

Generate results CSV from `logs/experiment_logs.txt`:
`python3 scripts/create_csv.py`

View tensorboard logs:
`python3 -m tensorboard.main --logdir=logs/runs`

# Sparse Text Generation

### Getting Started

Fetch training data:
`./utils/getdata.sh`

Before running the code, you need to install the dependencies by running the following lines:
```
cd language_modeling
pip3 install .
```
Install further requirements:
`pip3 install -r sparse_requirements.txt`

### Fine-tune GPT2 for Language Modeling

### Training
To fine-tune GPT2 for language modelling you just need to run the following command, modifying the parameters as you wish.
```
python3 examples/run_lm_finetuning.py \
        --train_data_file=/path/to/dataset/train \
        --eval_data_file=/path/to/dataset/eval \
        --output_dir=/path/to/output \
        --model_type=gpt2 \
        --model_name_or_path=gpt2-medium \
        --block_size=512 \
        --do_train \
        --evaluate_during_training \
        --loss=entmax \
        --entmax_alpha=1.2 \
        --top_k=0 \
        --top_p=0
```

### Evaluating
To evaluate a model just run:
```
python3 examples/run_lm_finetuning.py \
        --train_data_file=/path/to/dataset/train \
        --eval_data_file=/path/to/dataset/eval \
        --output_dir=/path/to/output \
        --model_type=gpt2 \
        --model_name_or_path=/path/to/checkpoint_to_evaluate \
        --block_size=512 \
        --do_eval \
        --loss=entmax \
        --entmax_alpha=1.2 \
        --top_k=0 \
        --top_p=0
```

### Fine-tune GPT2 for Dialogue Generation

### Training
To fine-tune GPT2 for dialogue generation you just need to run the following command, modifying the parameters as you wish.
```
python3 train.py 
        --dataset_path=/path/to/dataset \
        --model_checkpoint=gpt2-medium \
        --name=name_you_want_to_give_to_model \
        --loss=entmax \
        --entmax_alpha=1.3 \ 
        --top_p=0 \
        --top_k=0
```
### Evaluating
To evaluate a model just run:
```
python3 eval.py 
        --dataset_path=/path/to/dataset\
        --model_type=gpt2-medium \
        --name=name_you_want_to_give_to_model \
        --model_checkpoint=/path/to/checkpoint_to_evaluate \
        --loss=entmax \
        --entmax_alpha=1.3 \ 
        --top_p=0 \
        --top_k=0
```
A large portion of the Sparse Text Generation code comes from the awesome Huggingface [Transformers](https://github.com/huggingface/transformers) library and the [DeepSPIN](https://github.com/deep-spin/sparse_text_generation) research project coordinated by [Andre Martins](https://andre-martins.github.io/).

# Acknowledgements
Merity, S., Keskar, N.S. and Socher, R., 2017. Regularizing and optimizing LSTM language models. [arXiv preprint arXiv:1708.02182](https://arxiv.org/pdf/1708.02182.pdf).

Merity, S., Keskar, N.S. and Socher, R., 2018. An analysis of neural language modeling at multiple scales. [arXiv preprint arXiv:1803.08240](https://arxiv.org/pdf/1803.08240.pdf).

Mesham, S., Hayward, L., Shapiro, J. and Buys, J., 2021. Low-Resource Language Modelling of South African Languages. arXiv preprint [arXiv:2104.00772](https://arxiv.org/pdf/2104.00772.pdf).

Martins, P.H., Marinho, Z., Martins, A.F.T., 2020. Sparse Text Generation. arXiv preprint [arXiv:2004.02644](https://arxiv.org/abs/2004.02644).