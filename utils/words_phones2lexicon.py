#!/usr/bin/env python3


'''
Given words.txt and phones.txt generate lexicon.txt, for character-based situation only.
Special characters should begin with '<'
Usage: ./words_phones2lexicon.py /dir/to/words /dir/to/phones > lexicon.txt
'''

import sys


def read_phones(phones):
    all_phones = []
    with open(phones) as f:
        for line in f:
            phone = line.strip().split()[0]
            all_phones.append(phone)
    return all_phones


def read_words(words):
    all_words = []
    with open(words) as f:
        for line in f:
            word = line.strip().split()[0]
            all_words.append(word)
    return all_words


def main():
    words = sys.argv[1]
    phones = sys.argv[2]

    words = read_words(words)
    phones = read_phones(phones)

    print('<space> <space>')

    for word in words:
        if word.startswith('<'):
            if word in phones:
                print('%s %s' % (word, word))
            else:
                print('%s %s' % (word, '<unk>'))
        else:
            chars = list(word)
            for char in chars:
                assert char in phones
            print('%s %s' % (word, ' '.join(chars)))

    pass


if __name__ == "__main__":
    main()
