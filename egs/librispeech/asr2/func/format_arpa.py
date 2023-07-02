#!/usr/bin/env python3
import os
import sys



'''
Usage: python3 format_arpa.py lm_unpruned.arpa > lm_unpruned_new.arpa

This programme is basically used to solve the format problem of the n-gram LMs trained by Kaldi.

For example, in WSJ,

we will find that some lines in data/local/local_lm/4gram-mincount/lm_unpruned.arpa (obtained by gunzip -c lm_unpruned.gz > lm_unpruned.arpa) under \\1-grams for example,

float word float

where between float and word, word and float, there should be two \t symbols as below,

float\tword\tfloat

but by Kaldi there are <space> wrongly.

This programme corrects this error and reprint all lines in the original arpa file line by line.

This programme is written in a very dummy way and it supports at most 4gram.
'''

arpa = sys.argv[1]

stage = 0 # start
with open(arpa) as f:
    for line in f:
        lc = line.strip()
        items = lc.split()
        if stage == 0:
            if lc.startswith('\\1-grams'):
                stage = 1
            print(lc)
        elif stage == 1:
            if lc.startswith('\\2-grams'):
                stage = 2
                print(lc)
                continue
            if len(items) == 3:
                print('%.5f\t%s\t%.5f' % (float(items[0]), items[1], float(items[2])))
            elif len(items) == 2:
                print("%.5f\t%s" % (float(items[0]), items[1]))
            else:
                print(lc)
        elif stage == 2:
            if lc.startswith('\\3-grams'):
                stage = 3
                print(lc)
                continue
            if len(items) == 4:
                print('%.5f\t%s %s\t%.5f' % (float(items[0]), items[1], items[2], float(items[3])))
            elif len(items) == 3:
                print("%.5f\t%s %s" % (float(items[0]), items[1], items[2]))
            else:
                print(lc)
        elif stage == 3:
            if lc.startswith('\\4-grams'):
                stage = 4
                print(lc)
                continue
            if len(items) == 5:
                print('%.5f\t%s %s %s\t%.5f' % (float(items[0]), items[1], items[2], items[3], float(items[4])))
            elif len(items) == 4:
                print("%.5f\t%s %s %s" % (float(items[0]), items[1], items[2], items[3]))
            else:
                print(lc)
        elif stage == 4:
            if len(items) == 6:
                print('%.5f\t%s %s %s %s\t%.5f' % (float(items[0]), items[1], items[2], items[3], items[4], (items[5])))
            elif len(items) == 5:
                print("%.5f\t%s %s %s %s" % (float(items[0]), items[1], items[2], items[3], items[4]))
            else:
                print(lc)


