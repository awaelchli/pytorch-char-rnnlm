import codecs
import json

import torch


class Corpus(object):
    def __init__(self, path):
        _corpus = json.load(open(path))

        self.sents = []
        for s in _corpus:
            self.sents.append(list(s) + ['<eos>'])

    def export_char_list(self):
        s = set(c for s in self.sents for c in s)
        return list(sorted(list(s)))

    def tokenize(self, vocab, split=0.9):
        tokens = 0
        for s in self.sents:
            tokens += len(s)

        # pylint:disable=attribute-defined-outside-init
        self.ids = ids = torch.LongTensor(tokens)
        token = 0
        for s in self.sents:
            for w in s:
                ids[token] = vocab.char_to_idx[w]
                token += 1

        split_point = int(tokens * split)
        self.train_ids = ids[:split_point]
        self.eval_ids = ids[split_point:]


class Vocab(object):
    def __init__(self, char_list=None):
        if char_list is None:
            char_list = []

        self.char_list = list(char_list)
        self._build()

    def _build(self):
        char_list = self.char_list
        self.char_to_idx = {ch: i for (i, ch) in enumerate(char_list)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(char_list)}

    def size(self):
        return len(self.char_list)

    def save(self, filepath):
        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.char_list, f, indent=4, ensure_ascii=False)

    def load(self, filepath):
        self.char_list = json.load(open(filepath))
        self._build()

    @staticmethod
    def load(filepath):
        char_list = json.load(open(filepath))
        return Vocab(char_list)
