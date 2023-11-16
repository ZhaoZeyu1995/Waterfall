#!/usr/bin/env python3

import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from waterfall.utils.datapipe import Dataset, collate_fn_sorted, read_dict
from waterfall.utils.transforms import MaskWaveform
import argparse
import numpy as np
import logging
from waterfall import wav2vec
from kaldiio import WriteHelper
import time
from tqdm import tqdm


def predict(data_dir,
            lang_dir,
            model_dir,
            output_dir,
            jid,
            gpus=1,
            batch_size=1):
    '''
    data_dir, str, the data directory
    lang_dir, str, language directory
    model_dir, str, the path the of model parameters
    output_dir, str, the path where the output will be saved
    jid, int, the job id
    gpus: int, number of gpus for prediction, by default 1
    batch_size, int, by default 1
    '''

    model = wav2vec.Wav2VecModelNoWarmup.load_from_checkpoint(model_dir).cuda()

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'model'), 'w') as f:
        if os.path.isabs(model_dir):
            f.write(model_dir)
        else:
            f.write(os.path.join(os.getcwd(), model_dir))

    tic = time.time()
    masker = MaskWaveform()
    dataset = Dataset(data_dir, lang_dir, load_wav=True, transforms=masker)
    data_gen = DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn_sorted,
                          num_workers=4)
    logging.info('Predicting...')

    with open(os.path.join(output_dir, 'ref.wrd.trn.%d' % (jid)), 'w') as y:
        yc = ''
        with WriteHelper('ark,scp:%s,%s' % (os.path.join(os.getcwd(), output_dir, 'output.%d.ark' % (jid)), os.path.join(os.getcwd(), output_dir, 'output.%d.scp' % (jid)))) as writer:
            # Conduct prediction
            model.eval()
            for batch_idx, batch in tqdm(enumerate(data_gen)):
                # batch is a dict and contains the following keys: "wavs", "wav_lens", "targests", "names", "spks", "texts"
                # We need to put wavs and wav_lens on the gpu
                with torch.no_grad():
                    batch['wavs'] = batch['wavs'].cuda()
                    batch['wav_lens'] = batch['wav_lens'].cuda()
                    results = model.predict_step(batch, batch_idx)
                log_probs = results[0]
                log_probs = log_probs.cpu().detach().numpy()
                xlens = results[1]
                names = results[2]
                spks = results[3]
                texts = results[4]
                for i in range(log_probs.shape[0]):
                    single_probs = log_probs[i, :xlens[i], :]
                    writer(names[i], single_probs)

                    yc += '%s (%s-%s)\n' % (texts[i],
                                            spks[i], names[i])

        y.write(yc)
    toc = time.time()
    logging.info('Finished!')
    logging.info('Running time for job %d : %f' % (jid, toc-tic))


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description='Get the outputs (posterior probabilities) from a conformer model.')
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
