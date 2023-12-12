#!/usr/bin/env python3
import kaldiio
import numpy as np
import os
import argparse
from waterfall.utils.datapipe import read_dict


def main(args):
    prior_txt = read_dict(args.prior_dir)
    prior = np.zeros((len(prior_txt),))
    for key, value in prior_txt.items():
        prior[int(key)] = float(value)
    read_scp = os.path.join(args.predict_dir, "output.%d.scp" % (args.jid))

    os.makedirs(args.output_dir, exist_ok=True)
    write_ark = os.path.join(os.getcwd(), args.output_dir, "output.%d.ark" % (args.jid))
    write_scp = os.path.join(os.getcwd(), args.output_dir, "output.%d.scp" % (args.jid))

    with kaldiio.ReadHelper("scp:%s" % (read_scp)) as reader, kaldiio.WriteHelper(
        "ark,scp:%s,%s" % (write_ark, write_scp)
    ) as writer:
        for utt, post in reader:
            post = post / np.expand_dims(prior, axis=0)
            writer(utt, post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_dir", type=str)
    parser.add_argument("--predict_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--jid", type=int)

    args = parser.parse_args()

    main(args)
