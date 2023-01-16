#!/usr/bin/env python3

import sys

n_gram = sys.argv[1]

with open(n_gram) as f:
    count = 0
    flag = False
    for line in f:
        lc = line.strip().split()
        if flag and len(lc) >= 2:
            print(lc[1])
            continue
        if len(lc) > 0 and lc[0].endswith('1-grams:'):
            flag = True
        elif len(lc) > 0 and lc[0].endswith('2-grams:'):
            break

