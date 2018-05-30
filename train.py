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
    """Wraps hidden states in new Variables, to detach them from their history."""
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


def train_epoch(model, criterion, train_data, vocab, hps, lr, epoch, summary):
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = vocab.size()
    n_batches = len(train_data) // hps['bptt']
    hidden = model.init_hidden(hps['batch_size'])

    last_log_batch = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, hps['bptt'])):
        data, targets = get_batch(train_data, i, hps['bptt'])

        hidden = detach_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps['clip'])
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if (batch % hps['log_interval'] == 0 and batch > 0) or batch == n_batches:
            train_loss = loss.item()
            #train_loss = total_loss / (batch - last_log_batch)
            train_perp = math.exp(train_loss)

            last_log_batch = batch

            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:14.8f}'.format(
                    epoch, batch,
                    len(train_data) // hps['bptt'], lr,
                    elapsed * 1000 / hps['log_interval'], train_loss,
                   train_perp))

            step = hps['batch_size'] * helpers.get_num_batches(epoch, n_batches, batch)
            summary.add_scalar('TrainingLoss', train_loss, step)
            summary.add_scalar('TrainingPerp', train_perp, step)
            summary.add_scalar('GradientNorm', grad_norm, step)

            total_loss = 0
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

    # At any point you can hit Ctrl + C to break out of training early.
    print('-' * 95)
    try:
        for epoch in range(1, hps['epochs'] + 1):
            epoch_start_time = time.time()

            train_epoch(m, criterion, train_data, vocab, hps, lr, epoch, summary)

            with torch.no_grad():
                val_loss = evaluate(m, criterion, eval_data, vocab, hps)
                val_perp = math.exp(val_loss)
            print('-' * 95)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:14.8f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss,
                    val_perp))

            step = hps['batch_size'] * helpers.get_num_batches(epoch, n_batches, n_batches)
            summary.add_scalar('ValidationLoss', val_loss, step)
            summary.add_scalar('ValidationPerp', val_perp, step)

            if test_data is not None:
                with torch.no_grad():
                    test_loss = evaluate(m, criterion, test_data, vocab, hps)
                    test_perp = math.exp(test_loss)

                summary.add_scalar('TestLoss', test_loss, step)
                summary.add_scalar('TestPerp', test_perp, step)

                print(
                    '|                                 |  test loss {:5.2f} | '
                    ' test ppl {:14.8f}'.format(test_loss, test_perp))
            print('-' * 95)

            # Save the model if the validation loss is the best we've seen so
            # far.
            if not best_val_loss or val_loss < best_val_loss:
                os.makedirs(os.path.dirname(hps['save']), exist_ok=True)
                with open(hps['save'] + '.pt', 'wb') as f:
                    torch.save(m, f)

                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in
                # the validation dataset.
                lr /= 4

            summary.add_scalar('LR', lr, step)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def main():
    parser = argparse.ArgumentParser(description='PyTorch Language Model')
    parser.add_argument('hpsfile', type=str,
                        help='location of hyper parameter json file.')

    args = parser.parse_args()
    hps = json.load(open(args.hpsfile))
    device = torch.device(hps['device'] or 'cuda:0')
    # Remove log files when re-running
    shutil.rmtree(hps['log_dir'])

    summary = tensorboardX.SummaryWriter(hps['log_dir'])
    train(hps, device, summary)
    summary.export_scalars_to_json(os.path.join(hps['log_dir'], 'scalars.json'))


if __name__ == '__main__':
    main()
