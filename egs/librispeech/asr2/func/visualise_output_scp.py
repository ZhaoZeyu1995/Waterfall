#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import sys
from kaldiio import ReadHelper
import numpy as np

# usage example ./func/visualise_output_scp.py exp/ctc-blstmp/predict_train_yesno/output.1.scp data/lang_ctc/k2/tokens.txt

output_scp = sys.argv[1]
token_txt = sys.argv[2]

tokens = []
with open(token_txt) as f:
    for line in f:
        lc = line.strip().split()
        tokens.append(lc[0])

with ReadHelper('scp:%s' % (output_scp)) as reader:
    for uttid, log_prob in reader:
        prob = np.exp(log_prob)
        plt.figure()
        for idx in range(prob.shape[1]):
            plt.plot(prob[:, idx], label=tokens[idx])
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(output_scp), '%s.output.pdf' % (uttid)))
        plt.close()
