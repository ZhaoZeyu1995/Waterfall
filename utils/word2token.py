#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--non-lang-syms', default=None, type=str)
parser.add_argument('words', type=str)

def main(args):
    non_lang_syms = []
    if args.non_lang_syms:
        with open(args.non_lang_syms) as f:
            for line in f:
                non_lang_syms.append(line.strip().split()[0])

    with open(args.words) as f:
        for line in f:
            word = line.strip().split()[0]
            if word in non_lang_syms:
                print('%s %s' % (word, word))
            else:
                print(word, ' '.join(list(word)))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
