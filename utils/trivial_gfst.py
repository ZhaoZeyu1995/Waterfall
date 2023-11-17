#!/usr/bin/env python3
import sys
import re


'''
Generate a trivial text for compiling of G.fst
'''

with open(sys.argv[1]) as f:
    for line in f:
        word = line.strip().split()[0]
        # skip <*> and #digit
        if re.match(r'<.*>', word) or re.match(r'#\d+', word):
            continue
        print('0 0 %s %s 0.' % (word, word))
print('0')

