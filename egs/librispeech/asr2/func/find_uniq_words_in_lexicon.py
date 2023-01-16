#!/usr/bin/env python3

import sys

lexicon_words = sys.argv[1]
lm_words = sys.argv[2]

with open(lm_words) as f:
    lm_vocab = []
    for line in f:
        word = line.strip().split()[0]
        lm_vocab.append(word)

with open(lexicon_words) as f:
    for line in f:
        word = line.strip().split()[0]
        if word not in lm_vocab:
            print(word)
