#!/usr/bin/env python3
import kaldiio
import os
import numpy as np
from waterfall.utils.datapipe import Lang, read_dict, tokenise
import sys
import logging


'''

Takes as input

a text file dir
lang_dir
jid (optional for run.pl)

Outputs 

By default in the same dir as the text file, named as transition

This programme should usually be called by compute_transition_mat.sh 

'''


def target2transition_matrix(target: list,
                             dtype=np.float32):
    '''
    generate transition_matrix given a target (id_sequence)

    input
    target, list(int), target index sequence
    dtype

    return
    transition_matrix, torch.tensor with shape (S, S), where S = 2 * L + 1 for CTC 
        and L is the length of the target
    '''

    L = len(target)
    S = 2 * L + 1
    transition_matrix = np.full((S, S),
                                np.finfo(dtype).min,
                                dtype=dtype)
    for idx, item in enumerate(target):
        transition_matrix[2*idx, 2*idx] = 0.  # blank self loop
        if 2*idx - 1 >= 0:
            transition_matrix[2*idx, 2*idx - 1] = 0.  # last valid -> blank
        transition_matrix[2*idx+1, 2*idx+1] = 0.  # valid self loop
        transition_matrix[2*idx+1, 2*idx] = 0.  # last blank -> valid
        if 2*idx-1 >= 0 and target[idx] != target[idx-1]:
            transition_matrix[2*idx+1, 2*idx-1] = 0.  # last valid -> valid
    transition_matrix[S-1, S-1] = 0.  # final blank self loop
    transition_matrix[S-1, S-2] = 0.  # last valid -> final blank

    return transition_matrix


def main(text_dir, lang_dir, jid):
    logging.basicConfig(level=logging.INFO)
    utt2text = read_dict(text_dir)
    base_dir = os.path.dirname(os.path.abspath(text_dir))
    lang = Lang(lang_dir)
    with kaldiio.WriteHelper('ark,scp:%s,%s' % (os.path.join(base_dir, 'transition_mat.%d.ark' % (jid)), os.path.join(base_dir, 'transition_mat.%d.scp' % (jid)))) as writer:
        for utt, text in utt2text.items():
            target = tokenise(text, lang.token2idx)
            tm = target2transition_matrix(target)
            writer(utt, tm)
    logging.info('Generated transition_mats for %d utts' % (len(utt2text)))


if __name__ == "__main__":
    text_dir = sys.argv[1]
    lang_dir = sys.argv[2]
    if len(sys.argv) > 3:  # for run.pl
        jid = int(sys.argv[3])  # job id
    else:
        jid = 1
        logging.warning('Job ID is not specified and set to 1 by default.')
    main(text_dir, lang_dir, jid)
