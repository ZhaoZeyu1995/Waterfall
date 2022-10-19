#!/usr/bin/env python3

from kaldiio import ReadHelper
import argparse
import logging
import torch
import numpy as np
from torchaudio.models.decoder import ctc_decoder


def main(args):
    word_symbol_table = args.word_symbol_table
    words = []
    if word_symbol_table:
        with open(word_symbol_table) as f:
            for line in f:
                word = line.strip().split()[0]
                words.append(word)

    loglikeli_scp = args.log_likelihood
    decoder = ctc_decoder(lexicon=args.lexicon,
                tokens=args.tokens,
                lm=args.lm,
                beam_size=args.beam_size,
                beam_size_token=args.beam_size_token,
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                blank_token=args.blank_token,
                sil_token=args.sil_token,
                unk_word=args.unk_word)

    output_f = open(args.word_output, 'w')
    output_f_c = ''
    with ReadHelper('scp:%s' % (loglikeli_scp)) as reader:
        for utt, log_likelihood in reader:
            likelihood = torch.from_numpy(np.exp(np.expand_dims(log_likelihood, 0)))
            result = decoder(likelihood)
            print(result[0][0].words)
            exit()
            result_in_txt = ' '.join(list(map(str, result)))
            output_f_c += '%s %s\n' % (utt, result_in_txt)
            if words:
                word_sequence = ' '.join([words[idx] for idx in result])
            else:
                word_sequence = result_in_txt
            logging.info('%s %s' % (utt, word_sequence))
    output_f.write(output_f_c)
    output_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This programme conducts the token passing algorithm and decodes a given log_likelihood scp file.')

    parser.add_argument(
        'log_likelihood', help='The path of log_likelihood scp', type=str)
    parser.add_argument(
        'word_output', help='The path for saving output word sequences', type=str)
    parser.add_argument('lexicon', help='The lexicon file in the correct format. <word> <phone1> <phone2> ... <phone_sil>', type=str)
    parser.add_argument('tokens', help='The token file in the correct format. <token>', type=str)
    parser.add_argument('lm', help='The language model directory', type=str)

    parser.add_argument('--beam_size', help='The beam size for ctc decoder', type=int, default=50)
    parser.add_argument('--beam_size_token', help='The beam size for token', type=int, default=None)
    parser.add_argument('--beam_threshold', type=float, default=50)
    parser.add_argument('--lm_weight', type=float, default=2)
    parser.add_argument('--word_score', type=float, default=0)
    parser.add_argument('--blank_token', type=str, default='<blk>')
    parser.add_argument('--sil_token', type=str, default='SIL')
    parser.add_argument('--unk_word', type=str, default='<UNK>')
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
                        help='The path of the word symbol table, by which the output label ids can be translated into words', type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
