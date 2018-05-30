

def get_num_batches(epoch, num_batches, batch_idx):
    return (epoch - 1) * num_batches + batch_idx
