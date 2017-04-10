#!/usr/bin/env python3

import argparse
import json
import random

from bottle import route, run

import sample

parser = argparse.ArgumentParser(
    description='PyTorch Language Model (sampling, hosted as http server)')
parser.add_argument('--hps-file', type=str, required=True,
                    help='location of hyper parameter json file.')
parser.add_argument('--nb-tokens', type=int, default=100,
                    help='Number of characters to sample.')
parser.add_argument('--temperature', type=float, default=0.9,
                    help='Temperature for sampling. Higher values means higher diversity.')
parser.add_argument('--max-sents', type=int, default=1,
                    help='Maximal number of sentences, if there are at least so many.')
parser.add_argument('--host', type=str, default='localhost',
                    help='host to bind.')
parser.add_argument('--port', type=int, default=10000,
                    help='Port for serving sampling')
parser.add_argument('--prefix', type=str, default='sample',
                    help='prefix for accessing')


args = parser.parse_args()


global_blob = {}


@route('/' + args.prefix)
def f():
    args = global_blob['args']
    ctx = global_blob['ctx']
    hps = global_blob['hps']
    words = sample.sample(
        nb_tokens=args.nb_tokens,
        temperature=args.temperature,
        ctx=ctx
    )

    if hps['tokenization'] == 'word':
        sep = ' '
    elif hps['tokenization'] == 'char':
        sep = ''

    max_sents = random.randint(1, args.max_sents)
    sents = []
    sent = []

    for i, word in enumerate(words):
        if word == '<eos>' or i == len(words) - 1:
            sents.append(sent)
            sent = []
            if len(sents) >= max_sents:
                break
        sent.append(word)

    sents = [sep.join(sent) for sent in sents]

    puncts = set('，。！:!,.')
    sents = [
        sent + '' if (len(sent) > 0 and sent[-1] not in puncts) else sent
        for sent in sents
    ]

    result = ''.join(sents)
    data = {'sent': result}

    return json.dumps(data, ensure_ascii=False)


def main():
    global args
    hps = json.load(open(args.hps_file))
    hps['cuda'] = False  # force overloading

    ctx = sample.load(hps)

    # pylint:disable=global-statement
    global global_blob
    global_blob = {}
    global_blob['args'] = args
    global_blob['ctx'] = ctx
    global_blob['hps'] = hps

    run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
