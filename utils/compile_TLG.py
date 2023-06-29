#!/usr/bin/env python3
import k2
import kaldifst
import os
import sys
import logging
import torch
from waterfall.utils.datapipe import read_dict

def compile_TLG(lang_dir, determinize=True):
    """
    Compile the TLG (token-level graph) from the language model and lexicon.
    Args:
    lang_dir: The language directory, e.g., data/lang, which contains L_disambig.fst and G.fst and k2/T.fst

    Returns:
    HLG: The token-level graph, which is an FsaVec in k2.

    Note:
    The reason why I wrote this function is that I found the compatibility between k2 and OpenFST is not good.
    Currently, in most cases, we cannot load the lang/TLG.fst directly in k2 and use it to get the lattice.

    Also, this function is a modification of https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compile_hlg_using_openfst.py.
    """
    L = kaldifst.StdVectorFst.read(f"{lang_dir}/L_disambig.fst")
    logging.info("Arc sort L")
    kaldifst.arcsort(L, sort_type="olabel")
    logging.info(f"L: #states {L.num_states}")

    G_filename_binary = f"{lang_dir}/G.fst"
    logging.info(f"Loading {G_filename_binary}")
    G = kaldifst.StdVectorFst.read(G_filename_binary)


    logging.info("Arc sort G")
    kaldifst.arcsort(G, sort_type="ilabel")

    logging.info(f"G: #states {G.num_states}")

    logging.info("Compose L and G and connect LG")
    LG = kaldifst.compose(L, G, connect=True)
    logging.info(f"LG: #states {LG.num_states}")

    if determinize:
        logging.info("Determinizestar LG")
        kaldifst.determinize_star(LG)
        logging.info(f"LG after determinize_star: #states {LG.num_states}")

    logging.info("Minimize encoded LG")
    kaldifst.minimize_encoded(LG)
    logging.info(f"LG after minimize_encoded: #states {LG.num_states}")

    logging.info("Converting LG to k2 format")
    LG = k2.Fsa.from_openfst(LG.to_str(is_acceptor=False), acceptor=False)
    logging.info(f"LG in k2: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}")

    phone2id = read_dict(f"{lang_dir}/phones.txt")
    word2id = read_dict(f"{lang_dir}/words.txt")

    first_token_disambig_id = int(phone2id["#0"])
    first_word_disambig_id = int(word2id["#0"])
    logging.info(f"token id for #0: {first_token_disambig_id}")
    logging.info(f"word id for #0: {first_word_disambig_id}")

    T = kaldifst.StdVectorFst.read(f"{lang_dir}/k2/T.fst")
    T = k2.Fsa.from_openfst(T.to_str(is_acceptor=False), acceptor=False)
    logging.info(f"T: #states: {T.shape[0]}, #arcs: {T.num_arcs}")

    logging.info("Removing disambiguation symbols on LG")
    LG.labels[LG.labels >= first_token_disambig_id] = 0
    LG.aux_labels[LG.aux_labels >= first_word_disambig_id] = 0

    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    logging.info("Removing epsilons from LG")
    LG = k2.remove_epsilon(LG)
    logging.info(
        f"LG after k2.remove_epsilon: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}"
    )

    logging.info("Connecting LG after removing epsilons")
    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)
    logging.info(f"LG after k2.connect: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}")

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing T and LG")

    TLG = k2.compose(T, LG, inner_labels="phones")
    logging.info(
        f"TLG after k2.compose: #states: {TLG.shape[0]}, #arcs: {TLG.num_arcs}"
    )

    logging.info("Connecting TLG")
    TLG = k2.connect(TLG)
    logging.info(
        f"TLG after k2.connect: #states: {TLG.shape[0]}, #arcs: {TLG.num_arcs}"
    )

    logging.info("Arc sorting LG")
    TLG = k2.arc_sort(TLG)

    return TLG

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(filename)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    lang_dir = sys.argv[1]

    determinize = True
    if len(sys.argv) > 2:
        determinize = sys.argv[2]
        assert determinize in ["true", "false"], determinize
        determinize = True if determinize == "true" else False

    filename = f"{lang_dir}/TLG.pt"

    if os.path.exists(filename):
        logging.info(f"{filename} already exists - skipping")
    else:
        TLG = compile_TLG(lang_dir, determinize=determinize)
        logging.info(f"Saving TLG to {filename}")
        torch.save(TLG.as_dict(), filename)
