#!/usr/bin/env python3

import sys
import os

text = sys.argv[1]

all_words = set()
with open(text) as f:
    for line in f:
        fc = line.strip().split()
        words = fc[1:]
        all_words.update(words)
all_words = sorted(list(all_words))


dirname = os.path.dirname(text)
with open(os.path.join(dirname, 'vocab'), 'w') as f:
    fc = ''
    for word in all_words:
        fc += word + '\n'
    f.write(fc)
