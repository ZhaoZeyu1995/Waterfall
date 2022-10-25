#!/usr/bin/env python3

from kaldiio import ReadHelper
import argparse
import os
import logging
import torch
import numpy as np
import scipy
from waterfall.decoder import pyctcdecode


def main(args):
    words = []
    with open(os.path.join(args.lang, 'words.txt')) as f:
        for line in f:
            word = line.strip().split()[0]
            if word != "<eps>":
                words.append(word)

    tokens = []
    with open(os.path.join(args.lang, 'k2', 'tokens.txt')) as f:
        for line in f:
            token = line.strip().split()[0]
            tokens.append(token)

    loglikeli_scp = args.log_likelihood

    decoder = pyctcdecode.decoder_lexicon.build_ctcdecoder(
        lang=args.lang, labels=tokens, kenlm_model_path=args.lm, unigrams=words, alpha=args.alpha, beta=args.beta)


    output_f = open(args.word_output, 'w')
    output_f_c = ''
    count = 0
    with ReadHelper('scp:%s' % (loglikeli_scp)) as reader:
        for utt, log_likelihood in reader:
            text = decoder.decode(log_likelihood,
                                  beam_width=args.beam_width,
                                  beam_prune_logp=args.beam_prune_logp,
                                  token_min_logp=args.token_min_logp)
            output_f_c += '%s %s\n' % (utt, text.strip())
            logging.info('%s %s' % (utt, text.strip()))
            if count == 10:
                break
            count += 1
    output_f.write(output_f_c)
    output_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This programme conducts the token passing algorithm and decodes a given log_likelihood scp file.')

    parser.add_argument('lang', help='The lang directory.', type=str)
    parser.add_argument(
        'log_likelihood', help='The path of log_likelihood scp', type=str)
    parser.add_argument(
        'word_output', help='The path for saving output word sequences', type=str)
    # parser.add_argument('--tokens', help='The token file in the correct format. <token>', type=str, default='/disk/scratch3/zzhao/espnet/egs/librispeech/asr4/data/lang_char/train_960_unigram5000_units.txt')
    parser.add_argument('--lm', help='The language model directory',
                        type=str)

    parser.add_argument(
        '--alpha', help='The language model weight', type=float, default=0.5)
    parser.add_argument(
        '--beta', help='The word sequence length weight', type=float, default=1.5)
    parser.add_argument(
        '--beam_width', help='The maximum number of beams at each step in decoding', type=int, default=100)
    parser.add_argument(
        '--beam_prune_logp', help='Beams that are much worse than best beam will be pruned.', type=float, default=-10.0)
    parser.add_argument(
        '--token_min_logp', help='Tokens below this logp are skipped unless they are argmax of frame', type=float, default=-5.0)


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
