import re

if __name__ == '__main__':

    common_file = './data/penn/train_common1gram.txt'
    text_file = './data/penn/test.txt'
    output_file = './data/penn/test_compressed.txt'

    # Collect common words from file
    common_words = []
    with open(common_file, 'r') as f:
        for line in f:
            common_words += [line.strip('\r\n')]

    # Replace uncommon words in text
    with open(text_file, 'r') as text:
        with open(output_file, 'w') as output:
            for line in text:
                line_words = line.strip('\r\n').split(' ')

                new_line = line
                for w in line_words:
                    if w not in common_words:
                        pattern = r"\b" + w + r"\b"
                        new_line = re.sub(pattern, '<unk>', new_line)

                output.write(new_line)