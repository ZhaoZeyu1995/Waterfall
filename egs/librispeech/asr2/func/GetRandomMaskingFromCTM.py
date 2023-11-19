#!/usr/bin/env python3
"""
This script is used to get random masking from a CTM file.
Usage:
    GetRandomMaskingFromCTM.py <ctm_file> <output_file> <#masked words/utt>
    e.g. GetRandomMaskingFromCTM.py data/train/ctm data/train/mask_2 2
"""

import os
import sys
import random
import logging
import json


def load_ctm(ctm_file):
    ctm = {}
    with open(ctm_file, "r") as f:
        for line in f:
            line = line.strip().split()
            utt = line[0]
            start = float(line[2])
            end = start + float(line[3])
            word = line[4]
            if utt not in ctm:
                ctm[utt] = []
            ctm[utt].append((start, end, word))
    return ctm


def get_random_masking(ctm, num_masked):
    masking = {}
    for utt in ctm:
        masking[utt] = []
        words = ctm[utt]
        if len(words) < num_masked:
            logging.warning(
                "Number of words in utt {} is less than demanded {}.".format(
                    utt, num_masked
                )
            )
            for i in range(len(words)):
                masking[utt].append(words[i])
        else:
            random.shuffle(words)
            for i in range(num_masked):
                masking[utt].append(words[i])
    return masking


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ctm_file = sys.argv[1]
    output_file = sys.argv[2]
    num_masked = int(sys.argv[3])

    ctm = load_ctm(ctm_file)
    masking = get_random_masking(ctm, num_masked)

    with open(output_file, "w") as f:
        json.dump(masking, f, indent=4)
