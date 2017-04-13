#!/usr/bin/env python3

import argparse
import json

import torch
from torch.autograd import Variable

import data

# pylint:disable=redefined-builtin,no-member


def _load(hps):
    vocab = data.Vocab.load(hps['vocab_file'])
    if hps['cuda']:
        m = torch.load(hps['save'] + '.gpu')
    else:
        m = torch.load(hps['save'] + '.cpu')
    ntokens = vocab.size()
    return {'m': m, 'vocab': vocab, 'ntokens': ntokens, 'hps': hps}


class Sampler(object):
    def __init__(self, sampler_hps):
        self.sampler_hps = sampler_hps

        self.load()

    def load(self):
        sampler_hps = self.sampler_hps

        hps = self.hps = json.load(open(sampler_hps['hps_file']))
        hps['cuda'] = False  # force using cpu

        self.ctx = _load(hps)

    def sample(self, nb_tokens=None, temperature=None, max_sents=None):
        sampler_hps = self.sampler_hps
        hps = self.hps
        ctx = self.ctx

        if nb_tokens is None:
            nb_tokens = sampler_hps.get('nb_tokens', 100)
        if temperature is None:
            temperature = sampler_hps.get('temperature', 1.0)
        if max_sents is None:
            max_sents = sampler_hps.get('max_sents', 1)

        stop_at = sampler_hps.get('stop_at', None)

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

            if stop_at:
                if word == stop_at:
                    count_stop_at += 1
                    if count_stop_at >= max_sents:
                        break

        return words

    def join(self, words, max_sents=None):
        sampler_hps = self.sampler_hps
        hps = self.hps

        if max_sents is None:
            max_sents = sampler_hps.get('max_sents', 1)

        # first merge words to sentences
        if hps['tokenization'] == 'word':
            sep = ' '
        elif hps['tokenization'] == 'char':
            sep = ''

        sents = []
        sent = []

        for i, word in enumerate(words):
            if word == '<eos>' or i == len(words) - 1:
                sents.append(sent)
                sent = []
                if len(sents) >= max_sents:
                    break
                continue
            sent.append(word)

        sents = [sep.join(sent) for sent in sents]

        # then merge sentences to a single string
        sents_join_char = sampler_hps.get('sents_join_char', '\n')

        result = sents_join_char.join(sents)

        return result


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Language Model (sampling)')
    parser.add_argument('--sampler-hps-file', type=str, required=True,
                        help='location of sampler\'s hyper parameter json file.')
    parser.add_argument('--nb-tokens', type=int,
                        help='Number of characters to sample.')
    parser.add_argument('--temperature', type=float,
                        help='Temperature for sampling. Higher values means higher diversity.')
    parser.add_argument('--max-sents', type=int,
                        help='Maximum number of sentences to generate.')

    args = parser.parse_args()
    sampler_hps = json.load(open(args.sampler_hps_file))
    sampler = Sampler(sampler_hps)

    words = sampler.sample(
        args.nb_tokens,
        args.temperature,
        args.max_sents,
    )
    sent = sampler.join(words)

    print('=' * 80)
    print(sent)
    print('=' * 80)


if __name__ == '__main__':
    main()
