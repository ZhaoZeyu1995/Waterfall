#!/usr/bin/env python3

from waterfall.wfst_decoder_prefix import WFSTDecoder
from kaldiio import ReadHelper
import argparse
import logging


def main(args):
    fst_path = args.graph
    word_symbol_table = args.word_symbol_table
    words = []
    if word_symbol_table:
        with open(word_symbol_table) as f:
            for line in f:
                word = line.strip().split()[0]
                words.append(word)

    loglikeli_scp = args.log_likelihood

    decoder = WFSTDecoder(fst_path,
                          acoustic_scale=args.acoustic_scale,
                          max_active=args.max_active,
                          min_active=args.min_active,
                          beam=args.beam,
                          count_based_threshold=args.count_based_threshold,
                          score_based_threshold=args.score_based_threshold,
                          ac_cost_threshold=args.ac_cost_threshold,
                          allow_partial=args.allow_partial,
                          beam_delta=args.beam_delta)

    output_f = open(args.word_output, 'w')
    output_f_c = ''
    with ReadHelper('scp:%s' % (loglikeli_scp)) as reader:
        for utt, log_likelihood in reader:
            decoder.decode(log_likelihood)
            result = decoder.get_best_path()
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
        'graph', help='The path of the decoding WFST graph', type=str)
    parser.add_argument(
        'log_likelihood', help='The path of log_likelihood scp', type=str)
    parser.add_argument(
        'word_output', help='The path for saving output word sequences', type=str)
    parser.add_argument(
        '--acoustic_scale', help='The acoustic_scale for the log_likelihood, default 0.1.', type=float, default=0.1)
    parser.add_argument(
        '--max_active', help='The maximum number of the active tokens during decoding, larger, slower but more accurate, default 2000.', type=int, default=2000)
    parser.add_argument(
        '--min_active', help='The minimum number of the active tokens during decoding, default 20.', type=int, default=20)
    parser.add_argument(
        '--beam', help='The beam size during decoding, larger, slower but more accurate, default 16', type=float, default=16)
    parser.add_argument(
        '--score_based_threshold', help='The score-based-threshold during merging and pruning. The cost higher than the best prefix by this threshold are pruned, larger, slower but more accurate, default 16', type=float, default=20)
    parser.add_argument(
        '--count_based_threshold', help='The count-based-threshold during merging and pruning. The maximum number of prefixes during decoding, larger, slower but more accurate, default 50', type=int, default=50)
    parser.add_argument(
        '--ac_cost_threshold', help='The acoustic model cost before scaling, larger, slower but more accurate, default 10', type=float, default=10)
    parser.add_argument(
        '--beam_delta', help='The delta_beam for adjusting the pruning cutoff, default 0.5', type=float, default=0.5)
    parser.add_argument('--word_symbol_table',
                        help='The path of the word symbol table, by which the output label ids can be translated into words', type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
