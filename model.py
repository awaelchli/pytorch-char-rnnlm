import torch
from torch.autograd import Variable


class RNNModel(torch.nn.Module):

    #pylint:disable=line-too-long
    def __init__(self, ntokens, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntokens, ninp)

        self.rnn = torch.nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = torch.nn.Linear(nhid, ntokens)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    #pylint:disable=redefined-builtin
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(
            output.size(0) * output.size(1), output.size(2)))
        #pylint:disable=line-too-long
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
