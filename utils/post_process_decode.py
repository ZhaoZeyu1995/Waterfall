#!/usr/bin/env python3

import sys
from waterfall.utils.datapipe import read_dict

utt2spk = read_dict(sys.argv[2])


def process_quote(lc):
    if "QUOTE" not in lc:
        return lc
    else:
        flag = 0
        for idx, item in enumerate(lc):
            if item == 'QUOTE':
                if flag == 0:
                    flag = 1
                    lc[idx] = '"QUOTE'
                elif flag == 1:
                    flag = 0
                    lc[idx] = '"UNQUOTE'
        return lc


with open(sys.argv[1]) as f:
    for line in f:
        lc = line.strip().split()
        utt = lc[0]
        text = ' '.join(lc[1:])
        spk = utt2spk[utt]
        print('%s (%s-%s)' % (text, spk, utt))
