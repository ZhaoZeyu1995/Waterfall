#!/usr/bin/env python3


'''
Read a arpa file and extract 1-grams (words).
Usage: ./read1gram.py /dir/to/arpa > words.txt
'''

import sys

arpa = sys.argv[1]

with open(arpa) as f:
    flag = 0
    for line in f:
        if flag != 1:
            if line.startswith('\\1-grams'):
                flag = 1
            else:
                continue
        else:
            if line.startswith('\\2-grams') or len(line.strip().split()) == 0:
                break
            else:
                word = line.strip().split()[1]
                if word == '</s>' or word == '<s>':
                    continue
                else:
                    print(word)
