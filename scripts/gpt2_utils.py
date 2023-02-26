# Written by Stuart Mesham (MSHSTU001)

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import accumulate
from math import log
from typing import List, Tuple
from uuid import uuid4

import torch
from filelock import FileLock
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data import SequentialSampler, DataLoader
from tqdm.auto import tqdm
from transformers import (
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    is_torch_tpu_available,
)

from layer_switching_gpt2 import LayerSwitchingGPT2Config, GPT2LayerSwitchingLMHeadModel

# Add Weights & Biases logging
import wandb
import torch.nn as nn
import math

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

CACHE_DIR = "caches"
TB_DIR = "logs/runs"
LOG_DIR = "logs"
MODEL_DIR = "model_saves"

logger = logging.getLogger(__name__)


class CachedTextDataset(Dataset):
    """
    Adapted from transformers.TextDataset
    A class representing a single dataset. The dataset may consist of multiple files.
    The supplied tokenizer is used to tokenize the input text files.
    The tokenized dataset is stored in the CACHE_DIR specified in the gpt2_utils module.
    """

    def __init__(
        self,
        tokenizer: GPT2TokenizerFast,
        dataset_files: [str],
        block_size: int,
        stride: int,
        shuffle: bool = False,
    ):

        self.examples = []  # array containing training examples of length block_size

        for dataset_file in dataset_files:
            assert os.path.isfile(dataset_file)

            block_size = block_size - tokenizer.num_special_tokens_to_add(
                pair=False
            )  # allow for padding tokens

            # ------------------------------START CUSTOM CODE----------------------------------
            # calculate name of cache file
            m = hashlib.md5()
            m.update(tokenizer.cache_id.encode())
            m.update(str(block_size).encode())
            m.update(str(stride).encode())
            m.update(dataset_file.encode())
            cache_id = m.hexdigest()
            # -------------------------------END CUSTOM CODE-----------------------------------

            cached_features_file = os.path.join(
                CACHE_DIR,
                "dataset_{}".format(cache_id),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file):
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.examples += pickle.load(handle)
                    logger.info(
                        f"Loaded features from {cached_features_file} [took %.3f s]",
                        time.time() - start,
                    )

                else:
                    start = time.time()

                    with open(dataset_file, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(text)
                    )

                    dataset_examples = []  # training examples from this input file

                    # split tokenized_text into many examples of length block_size
                    for i in range(
                        0, len(tokenized_text) - block_size + 1, stride
                    ):  # Truncate in block of block_size
                        tokens = tokenizer.build_inputs_with_special_tokens(
                            tokenized_text[i : i + block_size]
                        )
                        dataset_examples.append(tokens)
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should look for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    # save cache
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(
                            dataset_examples, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )
                    logger.info(
                        f"Created features for {dataset_file} [took %.3f s]",
                        time.time() - start,
                    )

                    self.examples += dataset_examples

        if shuffle:
            self.examples = [
                self.examples[i] for i in torch.randperm(len(self.examples)).tolist()
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class MultilingualCachedTextDataset(Dataset):
    """
    A class representing a combination of datasets of different languages or language families.
    Creates batches each containing examples of only one language, facilitating the use of language specific layers.
    """

    def __init__(
        self,
        datasets_info: List[Tuple[int, GPT2TokenizerFast, List[str]]],
        block_size: int,
        stride: int,
        batch_size: int,
        shuffle: bool = False,
    ):
        self.datasets = [
            CachedTextDataset(
                tokenizer, dataset_files, block_size, stride, shuffle=shuffle
            )
            for _, tokenizer, dataset_files in datasets_info
        ]

        self.batch_size = batch_size

        self.language_ids = [language_id for language_id, _, _ in datasets_info]

        # round down so we don't get smaller batches at the end
        self._lengths = [len(dataset) // batch_size for dataset in self.datasets]
        self._cumulative_lengths = list(accumulate(self._lengths))

        self.length = sum(self._lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """
        :param i: index
        :return: Batch i of training examples. Each batch contains only only language.
        """
        dataset_index = 0
        while self._cumulative_lengths[dataset_index] <= i:
            dataset_index += 1

        if dataset_index == 0:
            first_example_index = i * self.batch_size
        else:
            first_example_index = (
                i - self._cumulative_lengths[dataset_index - 1]
            ) * self.batch_size

        batch = self.datasets[dataset_index][
            first_example_index : first_example_index + self.batch_size
        ]

        return {"language": self.language_ids[dataset_index], "input_ids": batch}


@dataclass
class MultilingualDataCollatorForLanguageModeling:
    """
    Simplified from huggingface's DataCollatorForLanguageModeling
    Data collator used for language modeling.
    Collates batches of tensors, honoring their tokenizer's pad_token.
    """

    pad_token_id: int

    def __call__(self, batch: dict) -> dict:
        labels = batch["input_ids"].clone().detach()
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def get_tokenizer(train_data, vocab_size):
    """
    Trains and returns a byte-level BPE tokenizer.
    If a cached tokenizer with these parameters exists it is loaded instead of training a new tokenizer.
    :param train_data: list of dataset files
    :param vocab_size: BPE vocab size
    :return: GPT2TokenizerFast with the requested parameters.
    """
    assert (
        vocab_size >= 257
    ), "vocab size must cover all possible bytes and one special token"

    # calculate the name of the cached file
    m = hashlib.md5()
    m.update(str(vocab_size).encode())
    for file in train_data:
        m.update(file.encode())
    cache_id = m.hexdigest()

    cached_tokenizer_file = os.path.join(CACHE_DIR, "tokenizer_{}".format(cache_id))

    train_new_tokenizer = not os.path.exists(cached_tokenizer_file)
    if train_new_tokenizer:
        start = time.time()
        os.makedirs(cached_tokenizer_file)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            train_data,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            show_progress=False,
        )
        tokenizer.save_model(cached_tokenizer_file)
        logger.info(
            f"Trained tokenizer {cached_tokenizer_file} [took %.3f s]",
            time.time() - start,
        )

    start = time.time()
    tokenizer = GPT2TokenizerFast.from_pretrained(cached_tokenizer_file)
    tokenizer.cache_id = cache_id

    if not train_new_tokenizer:
        logger.info(
            f"Loaded tokenizer from {cached_tokenizer_file} [took %.3f s]",
            time.time() - start,
        )

    return tokenizer


# ---------- MODIFIED CODE ----------
import numpy as np
from torch.nn import CrossEntropyLoss
from scipy.stats import entropy


def compute_jsd(p, q, base=np.e):
    p, q = np.asarray(p.cpu()), np.asarray(q.cpu())
    p, q = p / p.sum(), q / q.sum()
    m = 1.0 / 2 * (p + q)
    ent = entropy(p, m, base=base) / 2.0 + entropy(q, m, base=base) / 2.0
    if ent == float("Inf"):
        ent = torch.log(torch.FloatTensor([2]))
    return ent


def compute_sp(p, target):
    p = np.asarray(p.cpu())
    return 1 - (0.5 * np.linalg.norm(p) ** 2 - p[target] + 0.5)


import torch
from tqdm import tqdm
import math


def evaluate(
    tokenizers,
    model,
    eval_data,
    input_block_size,
    stride,
    epsilon=1e-8,
    disable_tqdm=False,
):
    """
    Evaluate the epsilon-perplexity performance of a model on a test dataset.
    :param tokenizers: list of tokenizers, one for each language
    :param model: the model to be evaluated
    :param eval_data: list of evaluation datasets to test the model's performance on
    :param input_block_size: size of the input block used for prediction
    :param stride: number of tokens to advance the input block per forward pass of the model
    :param disable_tqdm: disable evaluation progress bar
    :return: metrics for each evaluation datasets
    """
    assert stride <= input_block_size
    for language_id, file_paths in eval_data:
        total_characters = 0
        if len(file_paths) > 1:
            logger.warning(
                f"You supplied multiple eval files for language {language_id}. Only the first one will be used."
            )
        with open(file_paths[0], "r") as f:
            test_set = f.read()
        total_characters += len(test_set)
        encodings = tokenizers[language_id](test_set, return_tensors="pt")

        # adapted from https://huggingface.co/transformers/perplexity.html
        max_length = model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        log_probs = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # get the logits for the last token in each sequence
                logits = outputs[1][..., :-1, :].contiguous()
                logits = logits.view(-1, logits.size(-1))

                # apply softmax to get the probabilities
                probs = torch.softmax(logits, dim=1)

                # add epsilon to all terms and renormalize
                probs = probs + epsilon
                sums = probs.sum(dim=1, keepdim=True)
                probs = probs / sums

                # calculate log probabilities and take the mean
                log_probs.append(torch.log(probs).mean().item())

        # calculate epsilon-perplexity
        avg_log_prob = sum(log_probs) / len(log_probs)
        eps_ppl = torch.exp(
            torch.tensor((avg_log_prob)) / (1 + epsilon * model.config.vocab_size)
        )

        sp = 0
        jsd = 0

    result = {
        "sp": sp,
        "jsd": jsd,
        "e-perplexity": eps_ppl,
    }

    print("e-perplexity:", eps_ppl)
    print("js:", jsd)
    print("sp;", sp)

    return result, jsd, eps_ppl, sp


# def evaluate(
#     tokenizers,
#     model,
#     eval_data,
#     input_block_size,
#     stride,
#     eval_batch_size,
#     epsilon=0.000001,
# ):
#     perp = 0.0
#     model.eval()

#     jsd = 0
#     sp = 0

#     assert stride <= input_block_size
#     for language_id, file_paths in eval_data:
#         total_tokens = 0
#         if len(file_paths) > 1:
#             logger.warning(
#                 f"You supplied multiple eval files for language {language_id}. Only the first one will be used."
#             )
#         with open(file_paths[0], "r") as f:
#             test_set = f.read()
#         total_tokens += len(tokenizers[language_id].tokenize(test_set))
#         encodings = tokenizers[language_id](test_set, return_tensors="pt")

#     eval_sampler = SequentialSampler(test_set)
#     eval_dataloader = DataLoader(
#         test_set, sampler=eval_sampler, batch_size=eval_batch_size
#     )

#     for batch in tqdm(range(1, encodings.input_ids.size(1), stride), desc="Evaluating"):
#         begin_loc = max(batch + stride - input_block_size, 0)
#         end_loc = batch + stride
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-stride] = -100

#         with torch.no_grad():
#             outputs = model(
#                 input_ids,
#                 labels=target_ids,
#             )

#             shift_logits = outputs[1][..., :-1, :].contiguous()
#             shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#             shift_labels = target_ids[..., 1:].contiguous().squeeze(0)

#             probs = torch.softmax(shift_logits, dim=1)
#             lprobs = probs

#             if len(probs[0].nonzero()) != len(probs[0]):
#                 probs = probs[:, :] + epsilon
#                 sums = [probs[i].sum().item() for i in range(probs.size(0))]
#                 probs = [probs[i] / sums[i] for i in range(len(sums))]

#                 probs = torch.stack(probs)

#             p = [
#                 probs[i, shift_labels.squeeze(0)[i].item()]
#                 for i in range(len(shift_labels.squeeze(0)))
#             ]
#             p = torch.stack(p)
#             perp += torch.log(p**-1).mean().item()

#             jsd_batch = []
#             labels = torch.zeros(len(shift_labels), shift_logits.size(-1))
#             for i in range(len(shift_labels)):
#                 labels[i, shift_labels[i]] = 1
#                 jsd_ = compute_jsd(lprobs[i], labels[i])
#                 jsd_batch.append(jsd_)

#             jsd_batch = torch.tensor(jsd_batch).mean()
#             jsd += jsd_batch

#             sp_batch = []
#             for i in range(len(shift_labels)):
#                 sp_batch.append(
#                     compute_sp(lprobs.squeeze(0)[i], shift_labels[i]).item()
#                 )

#             sp_batch = torch.tensor(sp_batch).mean()
#             sp += sp_batch

#             pred = torch.multinomial(lprobs, num_samples=1).squeeze(1).view(-1).tolist()

#     print("perp:", perp)
#     print("jsd:", jsd)
#     print("sp:", sp)
#     print("len(eval_dataloader):", len(eval_dataloader))

#     a = perp / len(eval_dataloader)
#     perplexity = torch.exp(torch.tensor(a))

#     jsd = jsd / len(eval_dataloader)
#     sp = sp / len(eval_dataloader)

#     result = {
#         "sp": sp,
#         "JSD": jsd,
#         "perplexity": perplexity,
#     }

#     print("perplexity:", perplexity)
#     print("js:", jsd)
#     print("sp;", sp)

#     return result, jsd, perplexity, sp


# def evaluate(
#     tokenizers,
#     model,
#     eval_data,
#     input_block_size,
#     stride,
#     disable_tqdm=False,
#     epsilon=0.000001,
# ):
#     """
#     :param tokenizers: dict of tokenizers
#     :param model: model to evaluate
#     :param eval_data: list of tuples (language_id, file_paths)
#     :param input_block_size: size of the input block
#     :param stride: stride for the sliding window
#     :param disable_tqdm: disable tqdm progress bar
#     :param epsilon: epsilon for numerical stability
#     :return: perplexity, jsd, sp
#     """

#     perp = 0.0
#     model.eval()
#     jsd = 0
#     sp = 0

#     assert stride <= input_block_size
#     for language_id, file_paths in eval_data:
#         total_tokens = 0
#         if len(file_paths) > 1:
#             logger.warning(
#                 f"You supplied multiple eval files for language {language_id}. Only the first one will be used."
#             )
#         with open(file_paths[0], "r") as f:
#             test_set = f.read()
#         total_tokens += len(tokenizers[language_id].tokenize(test_set))
#         encodings = tokenizers[language_id](test_set, return_tensors="pt")

#     for i in tqdm(
#         range(1, encodings.input_ids.size(1), stride),
#         desc="Evaluating",
#         disable=disable_tqdm,
#     ):

#         begin_loc = max(i + stride - input_block_size, 0)
#         end_loc = i + stride
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-stride] = -100

#         with torch.no_grad():
#             outputs = model(
#                 input_ids,
#                 labels=target_ids,
#             )

#             shift_logits = outputs[1][..., :-1, :].contiguous()
#             shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#             shift_labels = target_ids[..., 1:].contiguous().squeeze(0)

#             probs = torch.softmax(shift_logits, dim=1)
#             lprobs = probs

#             if len(probs[0].nonzero()) != len(probs[0]):
#                 probs = probs[:, :] + epsilon
#                 sums = [probs[i].sum().item() for i in range(probs.size(0))]
#                 probs = [probs[i] / sums[i] for i in range(len(sums))]

#                 probs = torch.stack(probs)

#             p = [
#                 probs[i, shift_labels.squeeze(0)[i].item()]
#                 for i in range(len(shift_labels.squeeze(0)))
#             ]
#             p = torch.stack(p)
#             perp += torch.log(p**-1).mean().item()

#             jsd_batch = []
#             labels = torch.zeros(len(shift_labels), shift_logits.size(-1))
#             for i in range(len(shift_labels)):
#                 labels[i, shift_labels[i]] = 1
#                 jsd_ = compute_jsd(lprobs[i], labels[i])
#                 jsd_batch.append(jsd_)

#             jsd_batch = torch.tensor(jsd_batch).mean()
#             jsd += jsd_batch

#             sp_batch = []
#             for i in range(len(shift_labels)):
#                 sp_batch.append(
#                     compute_sp(lprobs.squeeze(0)[i], shift_labels[i]).item()
#                 )

#             sp_batch = torch.tensor(sp_batch).mean()
#             sp += sp_batch

#     a = perp / total_tokens
#     perplexity = torch.exp(torch.tensor(a))

#     jsd = jsd / total_tokens
#     sp = sp / total_tokens

#     result = {
#         "sp": sp,
#         "JSD": jsd,
#         "perplexity": perplexity,
#     }

#     print("perplexity:", perplexity)
#     print("js:", jsd)
#     print("sp;", sp)

#     return result, jsd, perplexity, sp


# -------------- END MODIFIED CODE --------------


def get_gpt2_trainer(
    experiment_id,
    hparams: dict,
    tparams: dict,
    disable_tqdm=False,
    disable_prediction_tqdm=True,
    log_to_console=False,
    resume_checkpoint_dir=None,
):
    """
    Creates a Trainer object based on the supplied parameters.
    :param experiment_id: experiment ID for logging
    :param hparams: model hyperparameter dictionary
    :param tparams: training parameters dictionary
    :param disable_tqdm: whether or not to disable training tqdm progress bar
    :param disable_prediction_tqdm: whether or not to disable evaluation tqdm progress bar during training
    :param log_to_console: whether or not to print loss values to the console during training
    :param resume_checkpoint_dir:
    :return: A Trainer object with the requested properties.
    """

    assert "vocab_size" in hparams
    assert "train_data" in hparams
    assert "val_data" in hparams
    assert "test_data" in hparams
    assert "model_max_input_size" in hparams
    assert "pdrop" in hparams
    assert "d_model" in hparams
    assert "n_layers" in hparams
    assert "n_heads" in hparams
    assert "train_block_size" in hparams
    assert "train_stride" in hparams
    assert "val_block_size" in hparams
    assert "val_stride" in hparams
    assert "learning_rate" in hparams
    assert "batch_size" in hparams
    assert "n_language_specific_attention_layers" in hparams
    assert "n_languages" in hparams
    assert "language_specific_input_embeds" in hparams
    assert "language_specific_prediction_heads" in hparams
    assert "semantic_concepts" in hparams
    assert "language_specific_transformation" in hparams
    assert "inverse_language_specific_transformation" in hparams
    assert "tie_word_embeddings" in hparams

    assert "max_steps" in tparams
    assert "patience" in tparams
    assert "log_steps" in tparams
    assert "eval_steps" in tparams
    assert "save_steps" in tparams

    if resume_checkpoint_dir is not None:
        model = GPT2LayerSwitchingLMHeadModel.from_pretrained(resume_checkpoint_dir)
    else:
        config = LayerSwitchingGPT2Config(
            vocab_size=hparams["vocab_size"],
            n_positions=hparams["model_max_input_size"],
            n_ctx=hparams["model_max_input_size"],
            n_embd=hparams["d_model"],
            n_layer=hparams["n_layers"],
            n_head=hparams["n_heads"],
            n_language_specific_attention_layers=hparams[
                "n_language_specific_attention_layers"
            ],
            n_languages=hparams["n_languages"],
            language_specific_input_embeds=hparams["language_specific_input_embeds"],
            language_specific_prediction_heads=hparams[
                "language_specific_prediction_heads"
            ],
            semantic_concepts=hparams["semantic_concepts"],
            language_specific_transformation=hparams[
                "language_specific_transformation"
            ],
            inverse_language_specific_transformation=hparams[
                "inverse_language_specific_transformation"
            ],
            attn_pdrop=hparams["pdrop"],
            embd_pdrop=hparams["pdrop"],
            resid_pdrop=hparams["pdrop"],
            summary_first_dropout=hparams["pdrop"],
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
            tie_word_embeddings=hparams["tie_word_embeddings"],
        )

        model = GPT2LayerSwitchingLMHeadModel(config=config)

    hparams["total_trainable_parameters"] = model.num_parameters(only_trainable=True)

    if "tokenizer_language_datasets" in hparams and "tokenizer_dataset" in hparams:
        raise ValueError(
            "You cannot specify both tokenizer_language_datasets and tokenizer_dataset."
        )
    elif "tokenizer_language_datasets" in hparams:
        assert (
            hparams["language_specific_input_embeds"]
            and hparams["language_specific_prediction_heads"]
        )
        assert len(hparams["tokenizer_language_datasets"]) == len(hparams["train_data"])
        tokenizers = {
            language_id: get_tokenizer(train_files, hparams["vocab_size"])
            for language_id, train_files in hparams["tokenizer_language_datasets"]
        }
    elif "tokenizer_dataset" in hparams:
        # all languages use the same tokenizer
        tokenizer = get_tokenizer(hparams["tokenizer_dataset"], hparams["vocab_size"])
        tokenizers = {
            language_id: tokenizer for language_id, _ in hparams["train_data"]
        }
    else:
        raise ValueError(
            "You must specify either tokenizer_language_datasets or tokenizer_dataset."
        )

    train_dataset = MultilingualCachedTextDataset(
        [
            (language_id, tokenizers[language_id], dataset_files)
            for language_id, dataset_files in hparams["train_data"]
        ],
        block_size=hparams["train_block_size"],
        stride=hparams["train_stride"],
        batch_size=hparams["batch_size"],
        shuffle=True,
    )

    validation_dataset = MultilingualCachedTextDataset(
        [
            (language_id, tokenizers[language_id], dataset_files)
            for language_id, dataset_files in hparams["val_data"]
        ],
        block_size=hparams["val_block_size"],
        stride=hparams["val_stride"],
        batch_size=hparams["batch_size"],
        shuffle=False,
    )

    data_collator = MultilingualDataCollatorForLanguageModeling(
        list(tokenizers.values())[
            0
        ].pad_token_id  # the same pad_token_id is used for all tokenizers
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, experiment_id),
        logging_dir=os.path.join(TB_DIR, experiment_id),
        save_steps=tparams["save_steps"],
        max_steps=tparams["max_steps"],
        per_device_train_batch_size=1,
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        logging_steps=tparams["log_steps"],
        eval_steps=tparams["eval_steps"],
        patience=tparams["patience"],
        prediction_loss_only=True,
        evaluate_during_training=True,
        disable_train_tqdm=disable_tqdm,
        disable_prediction_tqdm=disable_prediction_tqdm,
        hparams=hparams,
        is_dataset_pre_batched=True,
    )

    trainer = Trainer(
        tokenizers=tokenizers,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        log_to_console=log_to_console,
    )

    return trainer


def run_experiment(
    hparams: dict,
    tparams: dict,
    eval_stride=64,
    disable_tqdm=True,
    disable_prediction_tqdm=False,
    log_to_console=True,
    experiment_id=None,
    resume_checkpoint_dir=None,
    log_file="experiment_logs.txt",
):
    """
    Trains evaluates and logs the evaluation results of a model with the specified hyper-parameters
    :param hparams: the hyper-parameters to test
    :param tparams: the training parameters to test
    :param eval_stride: the stride to use during evaluation
    :param disable_tqdm: whether or not to disable training progress bars
    :param disable_prediction_tqdm: whether or not to disable progress bars for evaluation during training
    :param log_to_console: whether or to print loss value to the console during training
    :param experiment_id: Experiemnt ID used for logging. If None, a random one will be assigned.
    :param resume_checkpoint_dir: the checkpoint directory of an experiment to resume
    :param log_file: the file to save model evaluation results for the experiment
    """
    wandb.init(project="transformer-combined", config={"max_steps": 750})
    wandb.config.update(hparams)
    wandb.config.update(tparams)

    if experiment_id is None:
        experiment_id = uuid4().hex  # create a random experiment ID

    trainer = get_gpt2_trainer(
        experiment_id,
        hparams,
        tparams,
        disable_tqdm,
        disable_prediction_tqdm,
        log_to_console,
        resume_checkpoint_dir,
    )
    trainer.train(model_path=resume_checkpoint_dir)
    val_metrics, val_jsd, val_perplexity, val_sp = evaluate(
        trainer.tokenizers,
        trainer.model,
        hparams["val_data"],
        input_block_size=hparams["train_block_size"],
        stride=eval_stride,
        disable_tqdm=disable_prediction_tqdm,
    )
    logger.info(repr(val_metrics))
    # 🐝 Log train metrics to wandb
    wandb.log(
        {
            "step": trainer.global_step,
            "val_metrics": val_metrics,
            "jsd": val_jsd,
            "sp": val_sp,
            "ppl": val_perplexity,
        }
    )

    # test_metrics, test_jsd, test_perplexity, test_sp = evaluate(
    #     trainer.tokenizers,
    #     trainer.model,
    #     hparams["test_data"],
    #     input_block_size=hparams["train_block_size"],
    #     stride=eval_stride,
    #     disable_tqdm=disable_prediction_tqdm,
    # )
    # logger.info(repr(test_metrics))
    # # 🐝 Log train metrics to wandb
    # wandb.log(
    #     {
    #         "step": trainer.global_step,
    #         "test_metrics": test_metrics,
    #         "jsd": test_jsd,
    #         "sp": test_sp,
    #         "ppl": test_perplexity,
    #     }
    # )

    log_data = {
        "id": experiment_id,
        "completion_time": datetime.now().strftime("%x %X"),
        "steps": trainer.global_step,
        "hparams": hparams,
        "tparams": tparams,
        "val_metrics": val_metrics,
        "jsd": val_jsd,
        "sp": val_sp,
        "ppl": val_perplexity,
        # "test_metrics": test_metrics,
        # "jsd": test_jsd,
        # "sp": test_sp,
        # "ppl": test_perplexity,
    }

    with open(os.path.join(LOG_DIR, log_file), "a") as logfile:
        logfile.write(repr(log_data) + "\n")

    wandb.finish()
