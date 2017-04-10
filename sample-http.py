#!/usr/bin/env python3

import argparse
import json

from bottle import route, run

import sample

global_blob = {}


@route('/sample')
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

    sents = []
    sent = []

    for i, word in enumerate(words):
        if word == '<eos>' or i == len(words) - 1:
            sents.append(sent)
            if len(sents) >= args.max_sents:
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

    args = parser.parse_args()
    hps = json.load(open(args.hps_file))
    hps['cuda'] = False  # force overloading

    ctx = sample.load(hps)
    ctx['stop_at'] = '<eos>'

    # pylint:disable=global-statement
    global global_blob
    global_blob = {}
    global_blob['args'] = args
    global_blob['ctx'] = ctx
    global_blob['hps'] = hps

    run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
