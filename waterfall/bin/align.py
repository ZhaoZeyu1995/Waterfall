#!/usr/bin/env python3

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from waterfall.utils.datapipe_manual_ctc import Dataset, collate_fn
import argparse
import numpy as np
import logging
from tqdm import tqdm
from waterfall import models
from waterfall.manual_ctc.prior import compute_align
from kaldiio import WriteHelper
import time


def align(data_dir,
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

    model = models.Wav2VecFineTuningDiverseAlign.load_from_checkpoint(model_dir)

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

    print('Aligning...')
    results = trainer.predict(model, data_gen)
    print('Finished!')

    with WriteHelper('ark,scp:%s,%s' % (os.path.join(os.getcwd(), output_dir, 'align.%d.ark' % (jid)), os.path.join(os.getcwd(), output_dir, 'align.%d.scp' % (jid)))) as writer:
        for item in results:
            log_gamma_norm = item[0]
            xlens = item[1]
            trans_lengths = item[2]
            names = item[3]
            align_prob = np.exp(log_gamma_norm.cpu().detach().numpy())

            for i in range(align_prob.shape[0]):
                writer(names[i], align_prob[i, :xlens[i], :trans_lengths[i]])

    toc = time.time()
    print('Running time for job %d : %f' % (jid, toc-tic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This programme is for alignment of CTC.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--lang_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--jid', type=int)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)

    args = parser.parse_args()

    align(data_dir=args.data_dir,
          lang_dir=args.lang_dir,
          model_dir=args.model_dir,
          output_dir=args.output_dir,
          jid=args.jid,
          gpus=args.gpus,
          batch_size=args.batch_size)
