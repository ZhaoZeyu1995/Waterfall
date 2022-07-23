#!/usr/bin/env python3
import kaldiio
import os
import numpy as np
from waterfall.utils.datapipe import Lang, read_dict, tokenise
import sys
import logging


def target2transform_matrix(target,
                            num_token):
    '''
    generate transform matrix for a given sequence of targets
    each row is a one-hot vector

    input
    target, lis(int), the given targets
    idx2token, dict, mapping from tokenid to token

    return
    transform_matrix, torch.tensor, with shape (S, C),
        where S is the length of auxiliary sequence and C is the dimension of the neural network output
    '''

    L = len(target)
    S = 2 * L + 1
    C = num_token
    transform_matrix = np.full((S, C),
                                  np.finfo(np.float32).min,
                                  dtype=np.float32)
    for idx, item in enumerate(target):
        transform_matrix[2 * idx, 0] = 0.  # blank
        transform_matrix[2 * idx + 1, item] = 0.  # char token
    transform_matrix[S - 1, 0] = 0. # the last blank 
    
    return transform_matrix


def main(text_dir, lang_dir, jid):
    logging.basicConfig(level=logging.INFO)
    utt2text = read_dict(text_dir)
    base_dir = os.path.dirname(os.path.abspath(text_dir))
    lang = Lang(lang_dir)
    with kaldiio.WriteHelper('ark,scp:%s,%s' % (os.path.join(base_dir, 'transform_mat.%d.ark' % (jid)), os.path.join(base_dir, 'transform_mat.%d.scp' % (jid)))) as writer:
        for utt, text in utt2text.items():
            target = tokenise(text, lang.token2idx)
            tm = target2transform_matrix(target, len(lang.token2idx))
            writer(utt, tm)
    logging.info('Generated transform_mats for %d utts' % (len(utt2text)))


if __name__ == "__main__":
    text_dir = sys.argv[1]
    lang_dir = sys.argv[2]
    if len(sys.argv) > 3:  # for run.pl
        jid = int(sys.argv[3])  # job id
    else:
        jid = 1
        logging.warning('Job ID is not specified and set to 1 by default.')
    main(text_dir, lang_dir, jid)
