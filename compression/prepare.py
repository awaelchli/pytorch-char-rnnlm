from collections import Counter


def read_text(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [l.strip('\r\n') for l in lines]
    lines = [l.split(' ') for l in lines]

    words = []
    for line in lines:
        words += line

    return words


def top(word_list, n):
    counts = Counter(word_list)
    total = sum(counts.values())

    _common = counts.most_common(n)
    common = [t for t, _ in _common]

    # compute relative frequencies
    probs = [c for _, c in _common]
    probs = [c / total for c in probs]

    return common, probs


def unique_words(word_list):
    return list(set(word_list))


def get_ngrams(word_list, n):
    ngrams = []
    for i in range(len(word_list) - (n - 1)):
        ngram = tuple([word_list[i + j] for j in range(n)])
        ngrams += [ngram]
    return ngrams


def save_ngram_list(ngram_list, path):
    with open(path, 'w') as f:
        for ngram in ngram_list:
            line = ' '.join([w for w in ngram])
            f.write(line)
            f.write('\n')


def write_list_to_file(items, path):
    with open(path, 'w') as f:
        for item in items:
            f.write(str(item))
            f.write('\n')




if __name__ == '__main__':
    words = read_text('./data/penn/train.txt')
    prefix = 'train_'
    n = 1000
    print('Number of unique words: ', len(unique_words(words)))

    # Save a list of the most common words
    top_words, prob1 = top(words, n)
    write_list_to_file(top_words, './data/penn/' + prefix + 'common1gram.txt')
    write_list_to_file(prob1, './data/penn/' + prefix + 'common1prob.txt')

    # Save a list of the most common 2-grams
    two_grams = get_ngrams(words, 2)
    top_2grams, prob2 = top(two_grams, n)
    save_ngram_list(top_2grams, './data/penn/' + prefix + 'common2gram.txt')
    write_list_to_file(prob2, './data/penn/' + prefix + 'common2prob.txt')

    # Save a list of the most common 3-grams
    three_grams = get_ngrams(words, 3)
    top_3grams, prob3 = top(three_grams, n)
    save_ngram_list(top_3grams, './data/penn/' + prefix + 'common3gram.txt')
    write_list_to_file(prob3, './data/penn/' + prefix + 'common3prob.txt')


