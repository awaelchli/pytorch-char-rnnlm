#!/usr/bin/env python3

import argparse
import json
import sys

import torch
from torch.autograd import Variable

import data

# pylint:disable=redefined-builtin,no-member


def load(hps):
    vocab = data.Vocab.load(hps['vocab_file'])
    if hps['cuda']:
        m = torch.load(hps['save'] + '.gpu')
    else:
        m = torch.load(hps['save'] + '.cpu')
    ntokens = vocab.size()
    return {'m': m, 'vocab': vocab, 'ntokens': ntokens, 'hps': hps}


def sample(nb_tokens, temperature, ctx):
    m = ctx['m']
    vocab = ctx['vocab']
    ntokens = ctx['ntokens']
    hps = ctx['hps']

    hidden = m.init_hidden(1)

    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    input.data.fill_(vocab.char_to_idx['<eos>'])
    if hps['cuda']:
        input.data = input.data.cuda()

    words = []
    count_stop_at = 0
    for _ in range(nb_tokens):
        output, hidden = m(input, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = vocab.idx_to_char[word_idx]

        words.append(word)

        if ctx.get('stop_at', None) is not None:
            if word == ctx['stop_at']:
                count_stop_at += 1
                if count_stop_at >= (ctx.get('max_stop_at', 1)):
                    break

    return words


def print_words(words, hps, fp=sys.stdout):
    print('=' * 80, file=fp)
    for i, word in enumerate(words):
        if word == '<eos>' or i == len(words) - 1:
            print(file=fp)
        else:
            if hps['tokenization'] == 'word':
                sep = ' '
            elif hps['tokenization'] == 'char':
                sep = ''
            print(word, end=sep, file=fp)
    print('=' * 80, file=fp)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Language Model (sampling)')
    parser.add_argument('--hps-file', type=str, required=True,
                        help='location of hyper parameter json file.')
    parser.add_argument('--nb-tokens', type=int, default=100,
                        help='Number of characters to sample.')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Temperature for sampling. Higher values means higher diversity.')

    args = parser.parse_args()
    hps = json.load(open(args.hps_file))
    hps['cuda'] = False

    ctx = load(hps)
    words = sample(nb_tokens=args.nb_tokens,
                   temperature=args.temperature, ctx=ctx)
    print_words(words, hps)


if __name__ == '__main__':
    main()
