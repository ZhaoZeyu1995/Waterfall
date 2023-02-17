#!/usr/bin/env python3

import sys


'''
Generate a 2-state token list given a phones.txt
Ignore <SIL>
Only one single token for <eps> and disambig symbols
'''

fread = sys.argv[1]

with open(fread) as f:
    for line in f:
        lc = line.strip().split()[0]
        if lc == '<eps>':
            print(lc)
        elif lc == '<SIL>':
            continue
        elif lc.startswith('#'):
            print(lc)
        else:
            print(lc+'_0')
            print(lc+'_1')
        
