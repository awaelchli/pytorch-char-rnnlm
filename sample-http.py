#!/usr/bin/env python3

import argparse
import json
import random
import os

from bottle import route, run

import sample
global_blob = {}

def json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)

@route('/sample/<signature>')
def f(signature):
    if signature not in global_blob:
        return json_dump({'success': False, 'sent': ''})

    sampler = global_blob[signature]

    max_sents = random.randint(1, sampler.sampler_hps['max_sents'],)
    words = sampler.sample(max_sents=max_sents)
    sent = sampler.join(words)

    data = {'sent': sent, 'success': True}
    return json_dump(data)


def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Language Model (sampling, hosted as http server)')
    parser.add_argument(
        '--sampler-hps-file',
        type=str, required=True,
        action='append',
        help='location of sampler\'s hyper parameter json file. Can be specified for many times.'
    )
    parser.add_argument('--host', type=str, default='localhost',
                        help='host to bind.')
    parser.add_argument('--port', type=int, default=10000,
                        help='Port for serving sampling')

    args = parser.parse_args()

    for this_sampler_hps_file in args.sampler_hps_file:
        sampler_hps = json.load(open(this_sampler_hps_file))
        sampler = sample.Sampler(sampler_hps)
        signature = os.path.basename(this_sampler_hps_file).split('.')[0]
        global_blob[signature] = sampler
        print('%s <- %s' % (signature, this_sampler_hps_file))

    run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
