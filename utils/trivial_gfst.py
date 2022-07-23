#!/usr/bin/env python3
import sys


'''
Generate a trivial text for compiling of G.fst
'''

with open(sys.argv[1]) as f:
    for line in f:
        word = line.strip().split()[0]
        print('0 0 %s %s 0.' % (word, word))
print('0')

