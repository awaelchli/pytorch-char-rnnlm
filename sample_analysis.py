from sample import Sampler
import argparse
import json
from compression.prepare import *
import matplotlib.pyplot as plt
import numpy as np
import math


def get_text_sample(hps, length, temperature):
    sampler = Sampler(hps)

    tokens = sampler.sample(
        length,
        temperature,
    )
    text = sampler.join(tokens)
    return text


def save_hists(orig_words, orig_prob, sample_words, sample_prob, save=None):
    y1 = np.array(orig_prob)
    y2 = np.zeros(len(orig_prob))
    for i, orig_word in enumerate(orig_words):
        if orig_word in sample_words:
            k = sample_words.index(orig_word)
            y2[i] = sample_prob[k]

    write_list_to_file(list(y1), save + '_train.txt')
    write_list_to_file(list(y2), save + '_test.txt')


def main():
    sampler_hps_file = './hps/compression/more-params/sample-3-layers.json'
    output = './runs/compressed-3-layers-more-par/'
    length = 200
    num_samples = 2000
    temp = 1.0
    n = 1000  # top n word analysis
    ignore = ['<unk>', '']

    target_1grams = './data/penn/train_common1gram.txt'
    target_1probs = './data/penn/train_common1prob.txt'
    target_2grams = './data/penn/train_common2gram.txt'
    target_2probs = './data/penn/train_common2prob.txt'
    target_3grams = './data/penn/train_common3gram.txt'
    target_3probs = './data/penn/train_common3prob.txt'

    # Load target word histogram
    _1grams = read_tuples_from_file(target_1grams)
    _1probs = [float(x) for x in read_list_from_file(target_1probs)]
    _2grams = read_tuples_from_file(target_2grams)
    _2probs = [float(x) for x in read_list_from_file(target_2probs)]
    _3grams = read_tuples_from_file(target_3grams)
    _3probs = [float(x) for x in read_list_from_file(target_3probs)]

    # Select only a subset of top n tokens
    _1grams = _1grams[:n]
    _1probs = _1probs[:n]
    _2grams = _2grams[:n]
    _2probs = _2probs[:n]
    _3grams = _3grams[:n]
    _3probs = _3probs[:n]

    # Write generated text to a file
    hps = json.load(open(sampler_hps_file))
    all_samples = []
    with open(output + 'samples.txt', 'w') as f:
        for _ in range(num_samples):
            text = get_text_sample(hps, length, temp)
            all_samples += [text]
            f.write(text + '\n')

    # Make the histogram
    sample_1gram_list = get_words_from_lines(all_samples)
    sample_1gram_list = [w for w in sample_1gram_list if w not in ignore]

    sample_1grams = get_ngrams(sample_1gram_list, 1)
    sample_2grams = get_ngrams(sample_1gram_list, 2)
    sample_3grams = get_ngrams(sample_1gram_list, 3)

    sample_1grams, sample_1probs = top(sample_1grams, n, ignore)
    sample_2grams, sample_2probs = top(sample_2grams, n)
    sample_3grams, sample_3probs = top(sample_3grams, n)

    save_hists(_1grams, _1probs, sample_1grams, sample_1probs,
               save=output + '1-grams')
    save_hists(_2grams, _2probs, sample_2grams, sample_2probs,
               save=output + '2-grams')
    save_hists(_3grams, _3probs, sample_3grams, sample_3probs,
               save=output + '3-grams')


if __name__ == '__main__':
    main()
