#!/usr/bin/env python3

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from waterfall.utils.datapipe import Dataset, collate_fn_sorted, read_dict
import argparse
import numpy as np
import logging
from waterfall import wav2vec
from kaldiio import WriteHelper
import time


def predict(data_dir, lang_dir, model_dir, output_dir, jid, gpus=1, batch_size=1):
    """
    data_dir, str, the data directory
    lang_dir, str, language directory
    model_dir, str, the path the of model parameters
    output_dir, str, the path where the output will be saved
    jid, int, the job id
    gpus: int, number of gpus for prediction, by default 1
    batch_size, int, by default 1
    """

    model_dirs = model_dir.split(",")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model"), "w") as f:
        fc = ""
        for model_dir in model_dirs:
            if os.path.isabs(model_dir):
                fc += model_dir + "\n"
            else:
                fc += os.path.join(os.getcwd(), model_dir) + "\n"
        f.write(fc)

    logging.info(f"loading model from {model_dir}, in total {len(model_dirs)} models")

    all_results = []
    for model_dir in model_dirs:
        logging.info(f"loading model from {model_dir}")

        model = wav2vec.Wav2VecModel.load_from_checkpoint(model_dir)

        trainer = pl.Trainer(gpus=gpus, logger=None)

        tic = time.time()

        data_gen = DataLoader(
            Dataset(data_dir, lang_dir, load_wav=True),
            batch_size=batch_size,
            collate_fn=collate_fn_sorted,
            num_workers=4,
        )
        logging.info("Predicting...")
        results = trainer.predict(model, data_gen)
        logging.info("Finished!")

        all_results.append(results)

    # average the results which is the first element in each item in all_results
    logging.info(f"averaging {len(all_results)} results")
    results = []
    if len(all_results) == 1:
        results = all_results[0]
    else:
        for i in range(len(all_results[0])):
            log_probs = torch.stack(
                [all_results[j][i][0] for j in range(len(all_results))], dim=0
            )
            log_probs = torch.logsumexp(log_probs, dim=0) - np.log(len(all_results))
            results.append(
                (
                    log_probs,
                    all_results[0][i][1],
                    all_results[0][i][2],
                    all_results[0][i][3],
                    all_results[0][i][4],
                )
            )

    logging.info("Saving results...")
    with open(os.path.join(output_dir, "ref.wrd.trn.%d" % (jid)), "w") as y:
        yc = ""
        with WriteHelper(
            "ark,scp:%s,%s"
            % (
                os.path.join(os.getcwd(), output_dir, "output.%d.ark" % (jid)),
                os.path.join(os.getcwd(), output_dir, "output.%d.scp" % (jid)),
            )
        ) as writer:
            for item in results:
                log_probs = item[0]
                probs = log_probs.cpu().detach().numpy()
                xlens = item[1]
                names = item[2]
                spks = item[3]
                texts = item[4]
                for i in range(probs.shape[0]):
                    single_probs = probs[i, : xlens[i], :]
                    writer(names[i], single_probs)

                    yc += "%s (%s-%s)\n" % (texts[i], spks[i], names[i])
        y.write(yc)
    toc = time.time()
    logging.info("Running time for job %d : %f" % (jid, toc - tic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the outputs (log posterior probabilities) from the model(s)."
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--lang_dir", type=str)
    parser.add_argument(
        "--model_dir", type=str, help="the path of the models, separated by comma"
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--jid", type=int)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    predict(
        data_dir=args.data_dir,
        lang_dir=args.lang_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        jid=args.jid,
        gpus=args.gpus,
        batch_size=args.batch_size,
    )
