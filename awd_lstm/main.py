import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os
import hashlib
import data
import sys
from datetime import datetime
from model import LSTMModel
from utils import batchify, get_batch, repackage_hidden, model_save, model_load

parser = argparse.ArgumentParser(description="PyTorch AWD-LSTM Language Model")
parser.add_argument(
    "--data", type=str, default="data/penn/", help="location of the data corpus"
)
parser.add_argument(
    "--model", type=str, default="LSTM", help="type of recurrent net (LSTM, QRNN, GRU)"
)
parser.add_argument("--emsize", type=int, default=400, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=1150, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=3, help="number of layers")
parser.add_argument("--lr", type=float, default=30, help="initial learning rate")
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=8000, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=80, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=70, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.4,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--dropouth",
    type=float,
    default=0.3,
    help="dropout for rnn layers (0 = no dropout)",
)
parser.add_argument(
    "--dropouti",
    type=float,
    default=0.65,
    help="dropout for input embedding layers (0 = no dropout)",
)
parser.add_argument(
    "--dropoute",
    type=float,
    default=0.1,
    help="dropout to remove words from embedding layer (0 = no dropout)",
)
parser.add_argument(
    "--wdrop",
    type=float,
    default=0.5,
    help="amount of weight dropout to apply to the RNN hidden to hidden matrix",
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--nonmono", type=int, default=5, help="random seed")
parser.add_argument("--cuda", action="store_false", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=200, metavar="N", help="report interval"
)
randomhash = "".join(str(time.time()).split("."))
parser.add_argument(
    "--save", type=str, default=randomhash + ".pt", help="path to save the final model"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=2,
    help="alpha L2 regularization on RNN activation (alpha = 0 means no regularization)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=1,
    help="beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)",
)
parser.add_argument(
    "--wdecay", type=float, default=1.2e-6, help="weight decay applied to all weights"
)
parser.add_argument("--resume", type=str, default="", help="path of model to resume")
parser.add_argument(
    "--optimizer", type=str, default="sgd", help="optimizer to use (sgd, adam)"
)
parser.add_argument(
    "--when",
    nargs="+",
    type=int,
    default=[-1],
    help="When (which epochs) to divide the learning rate by 10 - accepts multiple",
)
# ----------Written by Victoria Pedlar---------- #
parser.add_argument(
    "--save_history",
    type=str,
    default=randomhash + ".txt",
    help="path to save the log history",
)
# ----------------------------------------------- #
# ----------Written by Luc Hayward---------- #
parser.add_argument(
    "--vocab_size", default=5000, help="size of vocab ONLY IF using bpe", type=int
)
parser.add_argument(
    "--use_bpe", default=True, help="use huggingface byte level bpe tokenizer"
)
parser.add_argument(
    "--early_exit",
    default=False,
    help="Exit early from model training once valid_loss is not changing enough per run",
)
parser.add_argument(
    "--descriptive_name",
    default="",
    help="Descriptive tag to add to the tensorboard save details.",
)
parser.add_argument(
    "--log_hparams_only",
    default=False,
    help="Skip training and jump straight to logging validation score for hparams metrics",
)
parser.add_argument("--basic", default=False)
parser.add_argument(
    "--chpc",
    default=False,
    help="Changes the tensoboard logging for chpc logging (no google drive)",
)
parser.add_argument(
    "--tokenizer_data",
    default="",
    help="Used when taking a model trained on one domain (this) and run against a different domain",
)
args = parser.parse_args()
args.tied = True
run_name = (
    str(args.data).replace("/", "-")
    + "/"
    + args.model
    + "/"
    + datetime.now().strftime("%d|%H:%M")
    + "_"
    + args.descriptive_name
)
sargs = ""
for arg in vars(args):
    sargs += "{:<16}: {}  \n".format(str(arg), str(getattr(args, arg)))
# if not args.log_hparams_only: writer.add_text('args', sargs)
print(sargs)
# ----------------------------------------------- #
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load the dataset and make train, validation and test sets

fn = "corpus.{}.data".format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn) and len(args.tokenizer_data) == 0:
    print("Loading cached dataset...")
    corpus = torch.load(fn)
else:
    print("Producing dataset...")
    corpus = data.Corpus(args.data, args.vocab_size, args.use_bpe, args.tokenizer_data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

# Build the model and specify the loss function

ntokens = len(corpus.dictionary)
model = LSTMModel(
    num_tokens=ntokens,
    embed_size=args.emsize,
    output_size=ntokens,
    hidden_size=args.nhid,
    n_layers=args.nlayers,
    dropout=args.dropout,
    dropouth=args.dropouth,
    dropouti=args.dropouti,
    dropoute=args.dropoute,
    wdrop=args.wdrop,
    tie_weights=args.tied,
)
criterion = nn.CrossEntropyLoss()

# Train the model
# First define training and evaluation


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(test_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args)
        # Starting each batch, we detach the hidden state from how it was previously produced
        # If we didn't, the model would try backpropagating all the way to start of the dataset
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


# Do the actual training
# Directing print output to a .txt file
sys.stdout = open(args.save_history, "wt")

# Loop over epochs
lr = args.lr
best_val_loss = []
stored_losses = 100000000

# At any point you can hit Ctrl + C to break out of training early
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if "t0" in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]["ax"].clone()

            val_loss2 = evaluate(val_data)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f} | valid bpc {:8.3f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss2,
                    math.exp(val_loss2),
                    val_loss2 / math.log(2),
                )
            )
            print("-" * 89)

            if val_loss2 < best_val_loss:
                model_save(args.save)
                best_val_loss = val_loss

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far
            if val_loss < best_val_loss:
                model_save(args.save)
                best_val_loss = val_loss

            elif len(stored_losses) > args.nonmono and val_loss > min(
                stored_losses[: -args.nonmono]
            ):
                print("Switching to ASGD")
                optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.0)

            stored_losses.append(val_loss)

except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

# # Open the best saved model run it on the test data
# model_load(args.save)

# # Run on test data
# test_loss = evaluate(test_data)
# print("=" * 89)
# print(
#     "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
#         test_loss, math.exp(test_loss)
#     )
# )
# print("=" * 89)
