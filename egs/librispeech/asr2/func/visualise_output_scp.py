#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import sys
from kaldiio import ReadHelper
import numpy as np
import argparse

# usage example ./func/visualise_output_scp.py exp/ctc-blstmp/predict_train_yesno/output.1.scp data/lang_ctc/k2/tokens.txt

parser = argparse.ArgumentParser(description='Visualise the outputs (posterior probabilities)')
parser.add_argument('output_scp', type=str, help='The output scp file')
parser.add_argument('--number', type=int, default=10, help='The number of figures to show')
parser.add_argument('--token_txt', type=str, help='The token.txt for showing the legend.', default=None)
parser.add_argument('--legend', type=bool, help='Whether or not show the legend', default=False)

args = parser.parse_args()

tokens = []
if args.token_txt:
    with open(args.token_txt) as f:
        for line in f:
            lc = line.strip().split()
            tokens.append(lc[0])

count = 0
with ReadHelper('scp:%s' % (args.output_scp)) as reader:
    for uttid, log_prob in reader:
        is_blk = True
        prob = np.exp(log_prob)
        plt.figure()
        for idx in range(prob.shape[1]):
            if args.legend:
                if is_blk:
                    plt.plot(prob[:, idx], '--', label=tokens[idx])
                    is_blk = False
                else:
                    plt.plot(prob[:, idx], label=tokens[idx])
            else:
                if is_blk:
                    plt.plot(prob[:, idx], '--')
                    is_blk = False
                else:
                    plt.plot(prob[:, idx])
        plt.ylim(0.0, 1.0)
        if args.legend:
            plt.legend()
        save_dir = os.path.join(os.path.dirname(args.output_scp), 'outputs')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, '%s.output.pdf' % (uttid)))
        plt.savefig(os.path.join(save_dir, '%s.output.png' % (uttid)))
        plt.close()
        count += 1
        if count >= args.number:
            break
