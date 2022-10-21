#!/usr/bin/env python3

import sys
import os

result_dir = sys.argv[1]
best_wer = None
best_dir = None

for name in os.listdir(result_dir):
    if os.path.isdir(os.path.join(result_dir, name)):
        if os.path.exists(os.path.join(result_dir, name, 'results.wrd.txt')):
            with open(os.path.join(result_dir, name, 'results.wrd.txt')) as f:
                for line in f:
                    lc = line.strip().split()
                    if 'Sum/Avg' in lc:
                        wer = float(lc[-3])
                        if best_wer == None or wer < best_wer:
                            best_wer = wer
                            best_dir = os.path.join(result_dir, name)

print(best_wer)
print(best_dir)

