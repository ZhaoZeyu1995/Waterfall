#!/usr/bin/env python3
import os
import argparse
import logging
from kaldiio import WriteHelper, ReadHelper
from pyctcdecode import build_ctcdecoder
from waterfall.utils.datapipe import Lang, read_dict


def decode(predict_path,
           data_dir,
           lang_dir,
           jid,
           lm=None,
           alpha=0.30,
           prune=-20,
           beam_prune=-20,
           beam_size=25,
           beta=5):
    '''
    Here is the documentation.
    '''

    utt2spk = read_dict(os.path.join(data_dir, 'utt2spk'))
    lang = Lang(lang_dir)

    token2idx = lang.token2idx

    labels = list(token2idx.keys())
    labels[0] = ""  # blank to nothing
    labels[labels.index("<space>")] = " "  # substitute <space> to ' '
    dec = build_ctcdecoder(
        labels=labels, kenlm_model_path=lm, beta=beta, alpha=alpha)

    result_path = os.path.join(os.path.dirname(
        predict_path), 'alpha_%.3f_beta_%d_beamsize_%d_beamprune_%d' % (alpha, beta, beam_size, beam_prune))
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(os.getcwd(), result_path, 'hyp.wrd.trn.%d' % (jid)), 'w') as x:
        xc = ''
        with ReadHelper('scp:%s' % (os.path.join(os.getcwd(), predict_path, 'output.%d.scp' % (jid)))) as reader:
            for uttid, data in reader:
                result = dec.decode(
                    data, beam_width=beam_size, token_min_logp=prune, beam_prune_logp=beam_prune)
                xc += '%s (%s-%s)\n' % (result,
                                        utt2spk[uttid], uttid)
        x.write(xc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This programme is for decoding of CTC.')

    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--lang_dir', type=str)
    parser.add_argument('--jid', type=int)
    parser.add_argument('--prune', type=float)
    parser.add_argument('--beam_prune', type=float)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--arpa_lm', type=str, default=None)

    args = parser.parse_args()

    decode(predict_path=args.predict_path,
           data_dir=args.data_dir,
           lang_dir=args.lang_dir,
           jid=args.jid,
           prune=args.prune,
           beam_prune=args.beam_prune,
           lm=args.arpa_lm,
           alpha=args.alpha,
           beam_size=args.beam_size,
           beta=args.beta)
