#!/usr/bin/env python3

import argparse
import json

import torch
import data


class Sampler(object):

    def __init__(self, sampler_hps):
        self.sampler_hps = sampler_hps
        self.train_hps = json.load(open(self.sampler_hps['hps_file']))
        self.device = torch.device(self.train_hps['device'])
        self._load()

    def _load(self):
        self.vocab = data.Vocab.load(self.train_hps['vocab_file'])
        self.ntokens = self.vocab.size()
        self.model = torch.load(self.train_hps['save'])

    def sample(self, length=None, temperature=None):
        sampler_hps = self.sampler_hps

        if length is None:
            length = sampler_hps.get('length', 100)
        if temperature is None:
            temperature = sampler_hps.get('temperature', 1.0)

        model = self.model
        vocab = self.vocab
        ntokens = self.ntokens

        hidden = model.init_hidden(1)

        input = torch.rand(1, 1).mul(ntokens).long()
        input.fill_(vocab.char_to_idx['<eos>'])
        input = input.to(self.device)

        chars = []

        for _ in range(length):
            with torch.no_grad():
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(temperature).exp()
                word_idx = torch.multinomial(word_weights, 1).item()
                input.fill_(word_idx)

            char = vocab.idx_to_char[word_idx]
            chars.append(char)

        # Remove truncated word at the end of the text
        end = chars[::-1].index(' ')
        chars = chars[:-end]
        return chars

    def join(self, tokens):
        sampler_hps = self.sampler_hps
        train_hps = self.train_hps

        # first merge words to sentences
        if train_hps['tokenization'] == 'word':
            sep = ' '
        elif train_hps['tokenization'] == 'char':
            sep = ''

        sents = []
        sent = []

        for i, word in enumerate(tokens):
            if word == '<eos>' or i == len(tokens) - 1:
                sents.append(sent)
                sent = []
                continue
            sent.append(word)

        sents = [sep.join(sent) for sent in sents]

        # then merge sentences to a single string
        sents_join_char = sampler_hps.get('sents_join_char', '\n')
        result = sents_join_char.join(sents)

        return result


def main():
    parser = argparse.ArgumentParser(description='PyTorch Language Model (sampling)')
    parser.add_argument('sampler_hps_file', type=str,
                        help='location of sampler\'s hyper parameter json file.')
    parser.add_argument('--length', type=int,
                        help='Number of characters to sample.')
    parser.add_argument('--temp', type=float,
                        help='Temperature for sampling. Higher values means higher diversity.')

    args = parser.parse_args()
    hps = json.load(open(args.sampler_hps_file))
    sampler = Sampler(hps)

    tokens = sampler.sample(
        args.length,
        args.temp,
    )
    text = sampler.join(tokens)

    print('=' * 80)
    print(text)
    print('=' * 80)


if __name__ == '__main__':
    main()
