

def get_num_batches_seen(epoch, num_batches, batch_idx):
    return (epoch - 1) * num_batches + batch_idx


def log_training(epoch, batch, num_batches, lr, ms_batch, train_loss, train_perp):
    print(
        f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | lr {lr:02.2f} '
        f'| ms/batch {ms_batch:5.2f} | loss {train_loss:5.2f} | ppl {train_perp:14.8f}')


def log_end_of_epoch(epoch, elapsed, valid_loss, valid_perp, test_loss, test_perp):
    print('-' * 95)
    print(
        f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s ' 
        f'| valid loss {valid_loss:5.2f} | ' f'valid ppl {valid_perp:14.8f} \n|' + ' ' * 33 +
        f'|  test loss {test_loss:5.2f} | 'f' test ppl {test_perp:14.8f}')
    print('-' * 95)
