#!/usr/bin/env python3

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from waterfall.utils.datapipe_k2 import Dataset, collate_fn, read_dict
import argparse
import numpy as np
import logging
from waterfall import models
from kaldiio import WriteHelper
import time


def predict(data_dir,
            lang_dir,
            model_dir,
            output_dir,
            jid,
            gpus=0,
            batch_size=1):
    '''
    wav_scp, str, wav.scp file
    lang_dir, str, lang directory
    model_dir, str, the path the of model parameters
    output_dir, str, the path where the output will be saved
    jid, int, the job id
    batch_size, int
    '''

    model = models.Wav2VecFineTuningDiverse.load_from_checkpoint(model_dir)

    trainer = pl.Trainer(gpus=gpus, logger=None)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'model'), 'w') as f:
        if os.path.isabs(model_dir):
            f.write(model_dir)
        else:
            f.write(os.path.join(os.getcwd(), model_dir))

    tic = time.time()
    data_gen = DataLoader(Dataset(data_dir, lang_dir),
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          num_workers=8)
    print('Predicting...')
    results = trainer.predict(model, data_gen)
    print('Finished!')

    with open(os.path.join(output_dir, 'ref.wrd.trn.%d' % (jid)), 'w') as y:
        yc = ''
        results = trainer.predict(model, data_gen)
        with WriteHelper('ark,scp:%s,%s' % (os.path.join(os.getcwd(), output_dir, 'output.%d.ark' % (jid)), os.path.join(os.getcwd(), output_dir, 'output.%d.scp' % (jid)))) as writer:
            for item in results:
                log_probs = item[0]
                probs = np.exp(log_probs.cpu().detach().numpy())
                xlens = item[1]
                names = item[2]
                spks = item[3]
                texts = item[4]
                for i in range(probs.shape[0]):
                    single_probs = probs[i, :xlens[i], :]
                    writer(names[i], single_probs)

                    yc += '%s (%s-%s)\n' % (texts[i],
                                            spks[i], names[i])
        y.write(yc)
    toc = time.time()
    print('Running time for job %d : %f' % (jid, toc-tic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This programme is for decoding of CTC.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--lang_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--jid', type=int)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()

    predict(data_dir=args.data_dir,
            lang_dir=args.lang_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            jid=args.jid,
            gpus=args.gpus,
            batch_size=args.batch_size)
