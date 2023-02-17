#!/usr/bin/env python3

import k2
import logging
import sys
import os
import torch

def main(lang_dir):
    print('Loading T.fst')
    with open(os.path.join(lang_dir, 'T.fst.txt')) as f:
        T = k2.Fsa.from_openfst(f.read(), acceptor=False)
    print('Loading L_disambig.fst')
    with open(os.path.join(lang_dir, 'L_disambig.fst.txt')) as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False)

    print('Loading G.fst')
    with open(os.path.join(lang_dir, 'G.fst.txt')) as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        # torch.save(G.as_dict(), "data/lm/G_3_gram.pt")

    
    print('arc_sort for L')
    L = k2.arc_sort(L)
    print('arc_sort for G')
    G = k2.arc_sort(G)

    logging.info("Composing L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    # LG.labels[LG.labels >= first_token_disambig_id] = 0

    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    # LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing T and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLG = k2.compose(T, LG, inner_labels="tokens")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"HLG.shape: {HLG.shape}")


    return HLG

if __name__ == "__main__":
    lang_dir = sys.argv[1]
    HLG = main(lang_dir)
    torch.save(HLG.as_dict(), os.path.join(lang_dir, 'HLG.fst.pt'))
