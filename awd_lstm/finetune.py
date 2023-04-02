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
from utils import batchify, get_batch, repackage_hidden, early_stopping
import wandb  # Add Weights & Bias logging

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
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=200, metavar="N", help="report interval"
)
randomhash = "".join(str(time.time()).split("."))
parser.add_argument(
    "--save", type=str, default=randomhash + ".pt", help="path to save the final model"
)
parser.add_argument(
    "--load", type=str, default=randomhash + ".pt", help="path to load pretrained model"
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
parser.add_argument(
    "-asgd",
    "--asgd",
    required=False,
    default="True",
    help="server on which this experiment runs",
)
# ----------Written by Victoria Pedlar---------- #
parser.add_argument(
    "--save_history",
    type=str,
    default=randomhash + ".txt",
    help="path to save the log history",
)
parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
parser.add_argument(
    "--patience", type=int, default=2, help="Patience for early stopping"
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

print("finetune load path: {}/model.pt. ".format(args.load))
print("log save path: {}/finetune_log.txt".format(args.save))
print("model save path: {}/finetune_model.pt".format(args.save))

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
# print(sargs)
# ----------------------------------------------- #
if torch.__version__ != "0.1.12_2":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

model_name = (
    "models/awd_lstm_finetune/"
    + "_emsize_"
    + str(args.emsize)
    + "_nhid_"
    + str(args.nhid)
    + "_nlayers_"
    + str(args.nlayers)
    + "_lr_"
    + str(args.lr)
    + "_wdc_"
    + str(args.wdecay)
    + "_clip_"
    + str(args.clip)
    + "_epochs_"
    + str(args.epochs)
    + "_bsz_"
    + str(args.batch_size)
    + "_bptt_"
    + str(args.bptt)
    + "_dropout_"
    + str(args.dropout)
    + "_dropouth_"
    + str(args.dropouth)
    + "_dropouti_"
    + str(args.dropouti)
    + "_dropoute_"
    + str(args.dropoute)
    + "_wdrop_"
    + str(args.wdrop)
    + "_seed_"
    + str(args.seed)
    + "_patience_"
    + str(args.patience)
    + "_when_"
    + str(args.when)
    + ".pt"
)
# ----------Written by Victoria Pedlar---------- #
log_every = 10
wandb.init(project="awd-lstm-finetuning", config={"lr": 50})
wandb.config.update(args)
config = wandb.config
# ----------------------------------------------- #
def model_load(file_name):
    """
    Loads the model and associated optimizer and criterion
    - Fixed the issue where cuda check is not performed causing crashes
    """
    global model, criterion, optimizer
    with open(file_name, "rb") as f:
        if torch.cuda.is_available():
            model, criterion, optimizer = torch.load(f)
        else:
            model, criterion, optimizer = torch.load(f, map_location="cpu")


###############################################################################
# Load data
###############################################################################

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

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model, criterion, optimizer = torch.load(args.load)
# model = LSTMModel(
#     num_tokens=ntokens,
#     embed_size=args.emsize,
#     output_size=ntokens,
#     hidden_size=args.nhid,
#     n_layers=args.nlayers,
#     dropout=args.dropout,
#     dropouth=args.dropouth,
#     dropouti=args.dropouti,
#     dropoute=args.dropoute,
#     wdrop=args.wdrop,
#     tie_weights=args.tied,
# )
# criterion = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()
    criterion = criterion.cuda()
total_params = sum(
    x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
    for x in model.parameters()
)

# Train the model
# First define training and evaluation
params = list(model.parameters()) + list(criterion.parameters())
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
total_params = sum(
    x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
    for x in params
    if x.size()
)


###############################################################################
# Training code
###############################################################################


# ---------- ADJUSTED CODE --------------
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


def evaluate(data_source, batch_size=10, epsilon=1e-8):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == "QRNN":
        model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    V = len(data_source)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)
            # Apply additive smoothing
            output += epsilon
            # Flatten the output and targets tensors
            output = output.view(-1, ntokens)
            targets = targets.view(-1)
            # Calculate cross-entropy loss
            loss = criterion(output, targets)
            total_loss += len(data) * loss.item()
            hidden = repackage_hidden(hidden)
        loss = total_loss / ((1 + epsilon) * V)
        eppl = math.exp(loss)
        sp_score = 0
        avg_jsd = 0
    return loss, eppl, avg_jsd, sp_score, loss / math.log(2)


# ------------- END ADJUSTED CODE --------------


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.0
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        raw_loss = criterion(output.view(-1, ntokens), targets)

        loss = raw_loss
        # Activation Regularization
        # if args.alpha:
        #     loss = loss + sum(
        #         args.alpha * dropped_rnn_h.pow(2).mean()
        #         for dropped_rnn_h in dropped_rnn_hs[-1:]
        #     )
        # # Temporal Activation Regularization (slowness)
        # if args.beta:
        #     loss = loss + sum(
        #         args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
        #         for rnn_h in rnn_hs[-1:]
        #     )
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]["lr"] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    optimizer.param_groups[0]["lr"],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                    cur_loss / math.log(2),
                )
            )
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

        if args.cuda:
            try:
                torch.cuda.empty_cache()
                # print('torch cuda empty cache')
            except:
                pass


# Load the best saved model.
# model_load(args.load)

# Do the actual training
# Directing print output to a .txt file
os.makedirs(os.path.dirname(args.save_history), exist_ok=True)
sys.stdout = open(args.save_history, "wt")

# Loop over epochs.
lr = args.lr
stored_loss, avg_perplexity, avg_jsd, avg_sp, bpc = evaluate(val_data)
best_val_loss = []
# early stopping parameter
stop_step = 0

print("Starting training...")
print(model_name)
for name, param in model.state_dict().items():
    print(name, param.size())

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, weight_decay=args.wdecay
        )  # params not trainable params... (?)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs + 1):
        print("Starting epoch {}".format(epoch))
        epoch_start_time = time.time()
        train()
        if args.cuda:
            try:
                torch.cuda.empty_cache()
                # print('torch cuda empty cache')
            except:
                pass
        if "t0" in optimizer.param_groups[0]:  # if ASGD
            tmp = {}
            for prm in model.parameters():
                if prm in optimizer.state.keys():
                    tmp[prm] = prm.data.detach()
                    prm.data = optimizer.state[prm]["ax"].detach()

            val_loss2, avg_perplexity, avg_jsd, avg_sp, bpc = evaluate(val_data)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss2,
                    math.exp(val_loss2),
                )
            )
            print("-" * 89)
            # 🐝 Log train metrics to wandb
            if (epoch + 1) % log_every == 0:  # subsampling
                wandb.log(
                    {
                        "loss": val_loss2,
                        "epoch": epoch,
                        "perplexity": avg_perplexity,
                        "JSD": avg_jsd,
                        "sp": avg_sp,
                        "bpc": bpc,
                    }
                )

            if val_loss2 < stored_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                print("Saving Averaged!")
                stored_loss = val_loss2

            for prm in model.parameters():
                # nparams += 1
                if prm in tmp.keys():
                    # nparams_in_temp_keys += 1
                    # prm.data = tmp[prm].clone()
                    prm.data = tmp[prm].detach()
                    prm.requires_grad = True
            # print('params {}, params in tmp keys: {}'.format(nparams, nparams_in_temp_keys))
            del tmp

            # begin early stopping
            if epoch % args.eval_every == (args.eval_every - 1):
                val_loss2, avg_perplexity, avg_jsd, avg_sp, bpc = evaluate(val_data)
                stored_loss, stop_step, stop = early_stopping(
                    val_loss2, stored_loss, stop_step, args.patience
                )
            if stop:
                break
            if stop_step == 0:
                best_epoch = epoch
                # model_save(args.save)
                with open(args.save, "wb") as f:
                    torch.save(model, f)

        else:
            print(
                "{} model params (SGD before eval)".format(
                    len([prm for prm in model.parameters()])
                )
            )
            val_loss, avg_perplexity, avg_jsd, avg_sp, bpc = evaluate(val_data)
            print(
                "{} model params (SGD after eval)".format(
                    len([prm for prm in model.parameters()])
                )
            )
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f} | valid bpc {:8.3f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                    val_loss / math.log(2),
                )
            )
            print("-" * 89)
            # 🐝 Log train metrics to wandb
            if (epoch + 1) % log_every == 0:  # subsampling
                wandb.log(
                    {
                        "loss": val_loss,
                        "epoch": epoch,
                        "perplexity": avg_perplexity,
                        "JSD": avg_jsd,
                        "sp": avg_sp,
                        "bpc": bpc,
                    }
                )

            if val_loss < stored_loss:
                # model_save(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                #            vocabulary, val_loss, math.exp(val_loss), vars(args), epoch)
                # model_save(args.save)
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                print("Saving model (new best validation)")
                stored_loss = val_loss

            if args.asgd:
                if (
                    args.optimizer == "sgd"
                    and "t0" not in optimizer.param_groups[0]
                    and (
                        len(best_val_loss) > args.nonmono
                        and val_loss > min(best_val_loss[: -args.nonmono])
                    )
                ):
                    # if 't0' not in optimizer.param_groups[0]:
                    print("Switching to ASGD")
                    # optimizer = ASGD(trainable_parameters, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    optimizer = torch.optim.ASGD(
                        trainable_parameters,
                        lr=args.lr,
                        t0=0,
                        lambd=0.0,
                        weight_decay=args.wdecay,
                    )

            if epoch in args.when:
                print("Saving model before learning rate decreased")
                # model_save('{}.e{}'.format(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                #            vocabulary, val_loss, math.exp(val_loss), vars(args), epoch))
                # model_save(args.save)
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                print("Dividing learning rate by 10")
                optimizer.param_groups[0]["lr"] /= 10.0

        best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

# # Load the best saved model.
# with open(args.save, "rb") as f:
#     model = torch.load(f)

# # Run on test data.
# test_loss = evaluate(test_data, test_batch_size)
# print("=" * 89)
# print(
#     "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
#         test_loss, math.exp(test_loss)
#     )
# )
# print("=" * 89)
# 🐝 Close your wandb run
wandb.finish()
