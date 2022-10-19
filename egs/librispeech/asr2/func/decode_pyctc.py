#!/usr/bin/env python3

from kaldiio import ReadHelper
import argparse
import logging
import torch
import numpy as np
import scipy
from waterfall.decoder import pyctcdecode


def main(args):
    word_symbol_table = args.word_symbol_table
    words = []
    if word_symbol_table:
        with open(word_symbol_table) as f:
            for line in f:
                word = line.strip().split()[0]
                if word != "<eps>":
                    words.append(word)

    tokens = ['']
    count = 1
    with open(args.tokens) as f:
        for line in f:
            token, number = line.strip().split()[0], line.strip().split()[1]

            # if token == '<blk>':
            # tokens.append(" ")
            # else:
            # print(count, number, token)
            count += 1
            tokens.append(token)

    loglikeli_scp = args.log_likelihood
    decoder = pyctcdecode.decoder.build_mmictcblkdecoder(
        labels=tokens, kenlm_model_path=args.lm, unigrams=words, alpha=args.alpha, beta=args.beta)
    print(len(tokens))

    logits = np.random.rand(15, len(tokens))
    exp_logits = np.exp(logits)
    log_likelihood = np.log(
        exp_logits / np.expand_dims(np.sum(exp_logits, axis=-1), axis=-1))
    text = decoder.decode(log_likelihood,
                          beam_width=args.beam_width,
                          beam_prune_logp=args.beam_prune_logp,
                          token_min_logp=args.token_min_logp)

    print(text)
    exit()

    # output_f = open(args.word_output, 'w')
    # output_f_c = ''
    # with ReadHelper('scp:%s' % (loglikeli_scp)) as reader:
    # for utt, log_likelihood in reader:
    # result = decoder(likelihood)
    # print(result[0][0].words)
    # exit()
    # result_in_txt = ' '.join(list(map(str, result)))
    # output_f_c += '%s %s\n' % (utt, result_in_txt)
    # if words:
    # word_sequence = ' '.join([words[idx] for idx in result])
    # else:
    # word_sequence = result_in_txt
    # logging.info('%s %s' % (utt, word_sequence))
    # output_f.write(output_f_c)
    # output_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This programme conducts the token passing algorithm and decodes a given log_likelihood scp file.')

    parser.add_argument(
        '--log_likelihood', help='The path of log_likelihood scp', type=str, default=None)
    parser.add_argument(
        '--word_output', help='The path for saving output word sequences', type=str, default='result.txt')
    # parser.add_argument('--tokens', help='The token file in the correct format. <token>', type=str, default='/disk/scratch3/zzhao/espnet/egs/librispeech/asr4/data/lang_char/train_960_unigram5000_units.txt')
    parser.add_argument('--tokens', help='The token file in the correct format. <token>',
                        type=str, default='data/lang_nosp_test_tgsmall_2state_blk/decode/tokens.txt')
    parser.add_argument('--lm', help='The language model directory',
                        type=str, default='data/local/lm/lm_tgsmall.arpa')

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

    # parser.add_argument(
    # '--acoustic_scale', help='The acoustic_scale for the log_likelihood, default 0.1.', type=float, default=0.1)
    # parser.add_argument(
    # '--max_active', help='The maximum number of the active tokens during decoding, larger, slower but more accurate, default 2000.', type=int, default=2000)
    # parser.add_argument(
    # '--min_active', help='The minimum number of the active tokens during decoding, default 20.', type=int, default=20)
    # parser.add_argument(
    # '--beam', help='The beam size during decoding, larger, slower but more accurate, default 16', type=float, default=16)
    # parser.add_argument(
    # '--beam_delta', help='The delta_beam for adjusting the pruning cutoff, default 0.5', type=float, default=0.5)
    parser.add_argument('--word_symbol_table',
                        help='The path of the word symbol table, by which the output label ids can be translated into words', type=str, default='data/lang_nosp/words.txt')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
