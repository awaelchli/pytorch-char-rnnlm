from compression.prepare import *
import matplotlib.pyplot as plt
import numpy as np
import math


def compare_hists(y1, y2, title='', save=None):
    # Create histogram plot
    x = np.arange(len(y1))

    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='test')
    plt.legend(loc='upper right')
    plt.title(title)
    if save:
        plt.savefig(save + '.eps', format='eps', dpi=1000)

    plt.show()

    # Compare numerically
    sum1 = sum(y1)
    sum2 = sum(y2)

    kl_div = 0
    for p, q in zip(y1, y2):
        q = 1e-8 if q == 0 else q
        kl_div += p * math.log2(p / q)

    # Save statistics to file
    with open(save + '.txt', 'w') as f:
        f.write('Area under curve "train": ' + str(sum1) + '\n')
        f.write('Area under curve "test":  ' + str(sum2) + '\n')
        f.write('KL-divergence:            ' + str(kl_div) + '\n')


if __name__ == '__main__':
    folder = '../runs/compressed-3-layers-more-par/'

    train_prob1 = folder + '1-grams_train.txt'
    test_prob1 = folder + '1-grams_test.txt'
    train_prob2 = folder + '2-grams_train.txt'
    test_prob2 = folder + '2-grams_test.txt'
    train_prob3 = folder + '3-grams_train.txt'
    test_prob3 = folder + '3-grams_test.txt'

    y1 = [float(x) for x in read_list_from_file(train_prob1)]
    y2 = [float(x) for x in read_list_from_file(test_prob1)]
    compare_hists(y1, y2, save=folder + '1-grams',
                  title='1-gram distribution most common')

    y1 = [float(x) for x in read_list_from_file(train_prob2)]
    y2 = [float(x) for x in read_list_from_file(test_prob2)]
    compare_hists(y1, y2, save=folder + '2-grams',
                  title='2-gram distribution most common')

    y1 = [float(x) for x in read_list_from_file(train_prob3)]
    y2 = [float(x) for x in read_list_from_file(test_prob3)]
    compare_hists(y1, y2, save=folder + '3-grams',
                  title='3-gram distribution most common')