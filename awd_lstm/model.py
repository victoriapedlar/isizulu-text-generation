import torch.nn as nn

from drop_connect import WeightDrop
from locked_dropout import LockedDropout
from embedding_dropout import embedded_dropout


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        hidden_size,
        embed_size,
        output_size,
        dropout=0.5,
        n_layers=1,
        wdrop=0,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        tie_weights=False,
    ):
        super(LSTMModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.tie_weights = tie_weights
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropout = dropout
        self.encoder = nn.Embedding(num_tokens, embed_size)

        # init LSTM layers
        self.lstms = []

        for l in range(n_layers):
            layer_input_size = embed_size if l == 0 else hidden_size
            layer_output_size = (
                hidden_size
                if l != n_layers - 1
                else (embed_size if tie_weights else hidden_size)
            )
            self.lstms.append(
                nn.LSTM(layer_input_size, layer_output_size, num_layers=1, dropout=0)
            )
        if wdrop:
            # Encapsulate lstms in DropConnect class to tap in on their forward() function and drop connections
            self.lstms = [
                WeightDrop(lstm, ["weight_hh_l0"], dropout=wdrop) for lstm in self.lstms
            ]
        self.lstms = nn.ModuleList(self.lstms)

        self.decoder = nn.Linear(
            embed_size if tie_weights else hidden_size, output_size
        )

        if tie_weights:
            # Tie weights
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        # Do embedding dropout
        emb = embedded_dropout(
            self.encoder, inp, dropout=self.dropoute if self.training else 0
        )
        # Do variational dropout
        emb = self.lockdrop(emb, self.dropouti)

        new_hidden = []
        outputs = []
        output = emb
        ## Remove RNN module weights not part of single contiguous chunk warning
        self.lstm.flatten_parameters()
        ##
        for i, lstm in enumerate(self.lstms):
            output, new_hid = lstm(output, hidden[i])

            new_hidden.append(new_hid)
            if i != self.n_layers - 1:
                # Do variational dropout
                output = self.lockdrop(output, self.dropouth)

        hidden = new_hidden
        # Do variational dropout
        output = self.lockdrop(output, self.dropout)

        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return [
            (
                weight.new(
                    1,
                    bsz,
                    self.hidden_size
                    if l != self.n_layers - 1
                    else (self.embed_size if self.tie_weights else self.hidden_size),
                ).zero_(),
                weight.new(
                    1,
                    bsz,
                    self.hidden_size
                    if l != self.n_layers - 1
                    else (self.embed_size if self.tie_weights else self.hidden_size),
                ).zero_(),
            )
            for l in range(self.n_layers)
        ]
