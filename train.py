#!/usr/bin/env python3
import tensorboardX
import argparse
import json
import math
import os
import time
import torch

import data
import model
import helpers
import shutil


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous().to(device)
    return data


def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(v.detach() for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, criterion, data_source, vocab, hps):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = vocab.size()
    hidden = model.init_hidden(hps['batch_size'])
    for i in range(0, data_source.size(0) - 1, hps['bptt']):
        data, targets = get_batch(data_source, i, hps['bptt'])
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).item()
        hidden = detach_hidden(hidden)
    return total_loss / len(data_source)


def train_epoch(model, criterion, train_data, vocab, hps, lr, epoch, device, summary):
    model.train()
    start_time = time.time()
    ntokens = vocab.size()
    n_batches = len(train_data) // hps['bptt']
    hidden = model.init_hidden(hps['batch_size'])

    for batch, i in enumerate(range(0, train_data.size(0) - 1, hps['bptt'])):
        data, targets = get_batch(train_data, i, hps['bptt'])

        hidden = detach_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps['clip'])

        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(-lr, p.grad.data)

        if (batch % hps['log_interval'] == 0 and batch > 0) or batch == n_batches:
            train_loss = loss.item()
            train_perp = math.exp(train_loss)

            elapsed = time.time() - start_time
            ms_per_batch = elapsed * 1000 / hps['log_interval']
            helpers.log_training(epoch, batch, n_batches, lr, ms_per_batch, train_loss, train_perp)

            step = hps['batch_size'] * helpers.get_num_batches_seen(epoch, n_batches, batch)
            summary.add_scalar('TrainingLoss', train_loss, step)
            summary.add_scalar('TrainingPerp', train_perp, step)
            summary.add_scalar('GradientNorm', grad_norm, step)

            start_time = time.time()


def train(hps, device, summary):
    train_corpus = data.Corpus(hps['train_corpus'], hps['tokenization'])
    eval_corpus = data.Corpus(hps['eval_corpus'], hps['tokenization'])
    test_corpus = None
    token_list = train_corpus.export_token_list() + eval_corpus.export_token_list()
    if hps['test_corpus']:
        test_corpus = data.Corpus(hps['test_corpus'], hps['tokenization'])
        token_list += test_corpus.export_token_list()

    vocab = data.Vocab(token_list)
    vocab.save(hps['vocab_file'])
    vocab = data.Vocab.load(hps['vocab_file'])

    train_corpus.tokenize(vocab)
    eval_corpus.tokenize(vocab)

    ntokens = vocab.size()

    m = model.RNNModel(
        ntokens,
        hps['emsize'],
        hps['nhid'],
        hps['nlayers'],
        hps['dropout'],
        hps['tied'],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    train_data = batchify(train_corpus.ids, hps['batch_size'], device)
    eval_data = batchify(eval_corpus.ids, hps['batch_size'], device)
    if test_corpus is not None:
        test_corpus.tokenize(vocab)
        test_data = batchify(test_corpus.ids, hps['batch_size'], device)
    else:
        test_data = None

    lr = hps['lr']
    best_val_loss = None
    n_batches = len(train_data) // hps['bptt']
    test_loss, test_perp = -1, -1

    # At any point you can hit Ctrl + C to break out of training early.
    print('-' * 95)
    try:
        for epoch in range(1, hps['epochs'] + 1):

            # Train for one epoch
            epoch_start_time = time.time()
            train_epoch(m, criterion, train_data, vocab, hps, lr, epoch, device, summary)
            elapsed = time.time() - epoch_start_time
            step = hps['batch_size'] * helpers.get_num_batches_seen(epoch, n_batches, n_batches)

            # Evaluate model on validation set
            with torch.no_grad():
                val_loss = evaluate(m, criterion, eval_data, vocab, hps)
                val_perp = math.exp(val_loss)

            summary.add_scalar('ValidationLoss', val_loss, step)
            summary.add_scalar('ValidationPerp', val_perp, step)

            # Evaluate model on test set
            if test_data is not None:
                with torch.no_grad():
                    test_loss = evaluate(m, criterion, test_data, vocab, hps)
                    test_perp = math.exp(test_loss)
                    test_bpc = test_loss * math.log2(math.e)

                summary.add_scalar('TestLoss', test_loss, step)
                summary.add_scalar('TestPerp', test_perp, step)
                summary.add_scalar('TestBPC', test_bpc, step)

            helpers.log_end_of_epoch(epoch, elapsed, val_loss, val_perp, test_loss, test_perp)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                os.makedirs(os.path.dirname(hps['save']), exist_ok=True)
                with open(hps['save'], 'wb') as f:
                    torch.save(m, f)

                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4

            summary.add_scalar('LR', lr, step)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def main():
    parser = argparse.ArgumentParser(description='PyTorch Language Model')
    parser.add_argument('hps_file', type=str,
                        help='location of hyper parameter json file.')

    args = parser.parse_args()
    hps = json.load(open(args.hps_file))
    device = torch.device(hps['device'] or 'cuda:0')

    # Remove log files when re-running
    if os.path.exists(hps['log_dir']):
        shutil.rmtree(hps['log_dir'])

    summary = tensorboardX.SummaryWriter(hps['log_dir'])
    train(hps, device, summary)
    summary.export_scalars_to_json(os.path.join(hps['log_dir'], 'scalars.json'))


if __name__ == '__main__':
    main()
