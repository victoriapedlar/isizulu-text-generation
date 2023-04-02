# Written by Stuart Mesham (MSHSTU001)

import sys
import logging
from gpt2_utils import run_experiment

# print logs to console
streamHandler = logging.StreamHandler(sys.stdout)
streamHandler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger = logging.getLogger("gpt2_utils")
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

tokenizer_train_data_zulu = [
    "data/isolezwe/isizulu/train.txt",
]

tokenizer_train_data_all = tokenizer_train_data_zulu
val_data = [
    (0, ["data/isolezwe/isizulu/valid.txt"]),
]

test_data = [
    (0, ["data/isolezwe/isizulu/test.txt"]),
]

hparams = {
    "tokenizer_dataset": tokenizer_train_data_zulu,
    "vocab_size": 8000,
    "train_data": [(0, tokenizer_train_data_zulu)],
    "val_data": val_data,
    "test_data": test_data,
    "model_max_input_size": 512,
    "pdrop": 0.1,
    "d_model": 16,
    "n_layers": 6,
    "n_heads": 4,
    "train_block_size": 128,
    "train_stride": 16,
    "val_block_size": 128,
    "val_stride": 128,
    "learning_rate": 1e-4,
    "batch_size": 16,
    "weight_decay": 0.1,
    "scheduler": "linear_with_warmup",
    "n_language_specific_attention_layers": 0,
    "n_languages": 1,
    "language_specific_input_embeds": False,
    "language_specific_prediction_heads": False,
    "language_specific_transformation": False,
    "inverse_language_specific_transformation": False,
    "semantic_concepts": None,
    "tie_word_embeddings": True,
}

tparams = {
    "max_steps": 1250,
    "patience": 2,
    "log_steps": 1,
    "eval_steps": 100,
    "save_steps": 100,
}

run_experiment(
    hparams,
    tparams,
    eval_stride=64,
    experiment_id="isolezwe_isizulu",
    resume_checkpoint_dir="model_saves/combined_isizulu/checkpoint-750",
)
