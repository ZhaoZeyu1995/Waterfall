#!/usr/bin/env python3
import kaldiio
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from waterfall.utils.datapipe import Lang, collate_fn, read_dict, tokenise


def get_single_prior(gamma: np.array, aux_seq: list):
    """
    args
    gamma: np.array, align matrix for one utt
    aux_seq: torch.tensor or list of int, the auxiliary sequence, e.g, [1, 2, 3] -> [0, 1, 0, 2, 0, 3, 0]

    return
    note, dict, note[token_id] = accumulated_prob
    """
    T, S = gamma.shape
    assert S == len(aux_seq)
    max_tokenid = np.argmax(gamma, axis=1)
    note = dict()  # token_id -> count
    for i in range(T):
        max_id = max_tokenid[i]
        token = aux_seq[max_id]
        if token in note.keys():
            note[token] += 1
        else:
            note[token] = 1
    return note


def get_aux_seq(tokens):
    aux_seq = [0] * (2 * len(tokens) + 1)
    for i, token in enumerate(tokens):
        aux_seq[2 * i + 1] = token
    return aux_seq


def main(args):
    utt2text = read_dict(os.path.join(args.data_dir, "text"))
    lang = Lang(args.lang_dir)
    accumulator = np.array([0.0] * len(lang.token2idx), dtype=np.float32)
    with kaldiio.ReadHelper("scp:%s" % (args.align_scp)) as align:
        for utt, gamma in tqdm(align):
            text = utt2text[utt]
            tokens = tokenise(text, lang.token2idx)
            aux_seq = get_aux_seq(tokens)
            note = get_single_prior(gamma, aux_seq)
            for key, value in note.items():
                accumulator[key] += value
    accumulator /= np.sum(accumulator)
    for idx, prior in enumerate(accumulator):
        if prior < 1e-6:
            prior = 1e-6
        print("%d %f" % (idx, prior))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--lang_dir", type=str)
    parser.add_argument("--align_scp", type=str)

    args = parser.parse_args()

    main(args)
