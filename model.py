import torch


class RNNModel(torch.nn.Module):

    def __init__(self, ntokens, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntokens, ninp)

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = torch.nn.Linear(nhid, ntokens)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.nhid = nhid
        self.nlayers = nlayers

        h0_params = torch.zeros(nlayers, 1, self.nhid)
        self.h0 = torch.nn.Parameter(h0_params)
        self.c0 = torch.nn.Parameter(h0_params.clone())

        # Do not learn initial states by default
        self.h0.requires_grad = False
        self.c0.requires_grad = False

        self.init_weights()

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(
            output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def learn_hidden_state(self, learn=True):
        self.h0.requires_grad = self.c0.requires_grad = learn

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        h0 = self.h0.repeat(1, bsz, 1)
        c0 = self.c0.repeat(1, bsz, 1)
        hidden = (h0, c0)
        return hidden
