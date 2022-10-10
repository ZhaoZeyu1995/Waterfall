#!/usr/bin/env python3
from kaldiio import ReadHelper
import numpy as np
import argparse
from tabulate import tabulate


def main(args):
    with open(args.token_list) as f:
        tokens = []
        for line in f:
            token = line.strip().split()[0]
            token_id = line.strip().split()[1]
            tokens.append((token, token_id))
    num_tokens = len(tokens)

    token2count = {}

    f = open(args.output, 'w')

    tot_max_token_soft = np.zeros((num_tokens,), dtype=np.float32)

    with ReadHelper('scp:%s' % (args.log_likelihood_scp)) as reader:
        tot_num_frames = 0
        for utt, log_likelihood in reader:
            num_frames = log_likelihood.shape[0]
            tot_num_frames += num_frames
            max_token = np.argmax(log_likelihood, axis=-1)
            max_token_soft = np.sum(np.exp(log_likelihood), axis=0)
            tot_max_token_soft += max_token_soft
            for i in range(num_tokens):
                num_i = num_frames - np.count_nonzero(max_token - i)
                if tokens[i] in token2count:
                    token2count[tokens[i]] += num_i
                else:
                    token2count[tokens[i]] = num_i

    tot_max_token_soft = tot_max_token_soft / tot_num_frames

    fc = tabulate([(token[0], token[1], count, count / tot_num_frames, tot_max_token_soft[idx]) for idx, (token, count)
                   in enumerate(token2count.items())] + [('sum', 'N/A', sum([count for _, count in token2count.items()]), sum([count / tot_num_frames for _, count in token2count.items()]), np.sum(tot_max_token_soft))], headers=('token', 'token_id', 'max_num', 'max_freq', 'mean_freq'), tablefmt='orgtbl')

    f.write(fc)
    f.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This programme calculates the frequency of each token which takes the maximum in the token set.')
    parser.add_argument('log_likelihood_scp',
                        help='The input log_likelihood_scp', type=str)
    parser.add_argument(
        'token_list', help='The token list without <eps>', type=str)
    parser.add_argument(
        'output', help='The output file with header, token, token_id, max_num, max_freq, mean_freq', type=str)
    args = parser.parse_args()
    main(args)
