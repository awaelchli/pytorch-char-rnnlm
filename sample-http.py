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
    first_sent_words = []
    for i, word in enumerate(words):
        if word == '<eos>' or i == len(words) - 1:
            break
        first_sent_words.append(word)
    sent = sep.join(first_sent_words)
    data = {'sent': sent}

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

    run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
