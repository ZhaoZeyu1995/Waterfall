#!/usr/bin/env python3

import sys

tokens = sys.argv[1]

with open(tokens) as f:
    for line in f:
        token = line.strip().split()[0]
        if token == '<blk>':
            print(token)
        elif token.endswith('_0'):
            print(token[:-2])
        elif token.endswith('_1'):
            print("%%" + token[:-2])
        else:
            print(token)

