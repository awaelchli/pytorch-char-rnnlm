import codecs
import json
import os

import torch


def _load_corpus(path):
    if path.endswith('.json'):
        corpus = json.load(open(path))
    else:
        corpus = [_.strip('\r\n') for _ in open(path).readlines()]
    return corpus


class Corpus(object):
    def __init__(self, path, tokenization='char'):
        assert tokenization in ['char', 'word']
        _corpus = _load_corpus(path)

        self.sents = []
        for s in _corpus:
            if tokenization == 'char':
                _s = list(s)
            elif tokenization == 'word':
                _s = s.split()
            self.sents.append(_s + ['<eos>'])

    def export_token_list(self):
        s = set(c for s in self.sents for c in s)
        return list(sorted(list(s)))

    def tokenize(self, vocab):
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


class Vocab(object):
    def __init__(self, token_list=None):
        if token_list is None:
            token_list = []

        token_list = list(sorted(list(set(token_list))))
        self.token_list = list(token_list)
        self._build()

    def _build(self):
        token_list = self.token_list
        self.char_to_idx = {ch: i for (i, ch) in enumerate(token_list)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(token_list)}

    def size(self):
        return len(self.token_list)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token_list, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load(filepath):
        token_list = json.load(open(filepath))
        return Vocab(token_list)
