import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import kaldiio
import json
import random
import logging
import torchaudio
import logging
from waterfall.utils.datapipe import Lang


'''
Data pipeline for kaldi data dir. Designated for k2.
'''

bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H

RATE = bundle.sample_rate


def read_list(file):
    '''
    Read a list file, e.g, phones.txt, words.txt, tokens.txt
    Return a list of all elements, where '<eps>' is always 0, which is a demand by openfst
    '''
    with open(file) as f:
        items = []
        for line in f:
            items.append(line.strip().split()[0])
    return items


def read_dict(file):
    '''
    Read a dict file, e.g, wav.scp, text, with the pattern <key> <value>
    Return a dict[<key>] = <value>
    '''
    a = dict()
    with open(file) as f:
        for line in f:
            lc = line.strip().split()
            a[lc[0]] = ' '.join(lc[1:])
    return a


def read_keys(file):
    '''
    Read a file with the pattern '<key> <value>'
    Return the keys in a list
    '''
    with open(file) as f:
        keys = []
        for line in f:
            utt = line.strip().split()[0]
            keys .append(utt)
    return keys


def tokenise(text, token2idx):
    '''
    tokenise a text given token2idx
    text: str
    token2idx: dict

    return a list of tokens
    '''
    words = text.split(' ')
    tokens = []
    for word in words:
        if word in token2idx.keys():
            tokens.append(token2idx[word])
        else:
            tokens.extend([token2idx[token] for token in list(word)])
        tokens.append(token2idx['<space>'])

    return tokens[:-1]  # Get rid of the last <space>


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 lang_dir,
                 token_type='tokens',
                 load_wav=True,
                 load_feats=False,
                 do_delta=False,
                 transforms=None):
        '''
        Read a Kaldi data dir, which should usually contain

        wav.scp
        utt2spk
        text
        spk2utt
        spk2gender

        TODO: This is no need to process feats.scp and cmvn.scp as 
        currently we only train our models based on wav2vec 2.0

        '''
        self.data_dir = data_dir
        self.lang_dir = lang_dir
        self.lang = Lang(self.lang_dir, token_type=token_type)

        self.load_wav = load_wav
        self.wav_scp = os.path.join(self.data_dir, 'wav.scp')
        self.utt2wav = kaldiio.load_scp(self.wav_scp)

        self.uttids = read_keys(self.wav_scp)
        self.utt2spk = read_dict(os.path.join(self.data_dir, 'utt2spk'))
        self.utt2text = read_dict(os.path.join(self.data_dir, 'text'))

        self.load_feats = load_feats
        if do_delta:
            self.dump_feats = os.path.join(self.data_dir, 'dump', 'deltatrue', 'feats.scp')
        else:
            self.dump_feats = os.path.join(self.data_dir, 'dump', 'deltafalse', 'feats.scp')
        self.utt2feats = kaldiio.load_scp(self.dump_feats)

        self.transforms = transforms

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        text = self.utt2text[uttid]
        target = []
        words = text.split(' ')
        word_ids = []
        for word in words:
            if word in self.lang.word2idx.keys():
                word_ids.append(self.lang.word2idx[word])
            else:
                # by default all unknown word should be denoted by <UNK>
                word_ids.append(self.lang.word2idx['<UNK>'])

        sample = {
            'target_length': len(target),
            'target': target,
            'name': uttid,
            'spk': spk,
            'word_ids': word_ids,
            'text': text}

        if self.load_wav:
            rate, wav = self.utt2wav[uttid]
            wav = torch.tensor(wav, dtype=torch.float32)
            if rate != RATE:
                wav = torchaudio.functional.resample(wav, rate, RATE)

            wav = (wav - wav.mean()) / torch.sqrt(wav.var())  # Normalisation
            sample['wav'] = wav
            sample['length'] = len(wav)

        if self.load_feats:
            feats = torch.tensor(self.utt2feats[uttid], dtype=torch.float32)
            sample['feats'] = feats
            sample['feats_len'] = feats.shape[0]

        if self.transforms:
            return self.transforms(sample)
        else:
            return sample


def collate_fn(list_of_samples):
    # Collect data
    batch_targets = [sample['target'] for sample in list_of_samples]
    batch_target_lengths = [sample['target_length']
                            for sample in list_of_samples]
    batch_names = [sample['name'] for sample in list_of_samples]
    batch_spks = [sample['spk'] for sample in list_of_samples]
    batch_texts = [sample['text'] for sample in list_of_samples]
    batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

    samples_collated = {'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                        'targets': batch_targets,
                        'names': batch_names,
                        'spks': batch_spks,
                        'texts': batch_texts}

    if 'wav' in list_of_samples[0].keys():
        batch_wav = [sample['wav'] for sample in list_of_samples]
        batch_lengths = [sample['length'] for sample in list_of_samples]
        samples_collated['wavs'] = torch.nn.utils.rnn.pad_sequence(
            batch_wav, batch_first=True)
        samples_collated['lengths'] = torch.tensor(
            batch_lengths, dtype=torch.long)

    if 'feats' in list_of_samples[0].keys():
        batch_feats = [sample['feats'] for sample in list_of_samples]
        batch_feats_lens = [sample['feats_len'] for sample in list_of_samples]
        samples_collated['feats'] = torch.nn.utils.rnn.pad_sequence(
            batch_feats, batch_first=True)
        samples_collated['feats_lens'] = torch.tensor(
            batch_feats_lens, dtype=torch.long)

    return samples_collated


def collate_fn_sorted(list_of_samples):
    # Collect data

    if 'wav' in list_of_samples[0].keys():
        batch_targets = [sample['target'] for sample in list_of_samples]
        batch_target_lengths = [sample['target_length']
                                for sample in list_of_samples]
        batch_names = [sample['name'] for sample in list_of_samples]
        batch_spks = [sample['spk'] for sample in list_of_samples]
        batch_texts = [sample['text'] for sample in list_of_samples]
        batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

        batch_wav = [sample['wav'] for sample in list_of_samples]
        batch_lengths = [sample['length'] for sample in list_of_samples]

        # Sorted
        batch_wav = [x for _, x in sorted(
            zip(batch_lengths, batch_wav), key=lambda x:x[0], reverse=True)]
        batch_targets = [x for _, x in sorted(
            zip(batch_lengths, batch_targets), key=lambda x:x[0], reverse=True)]
        batch_target_lengths = [x for _, x in sorted(
            zip(batch_lengths, batch_target_lengths), key=lambda x:x[0], reverse=True)]

        batch_names = [x for _, x in sorted(
            zip(batch_lengths, batch_names), key=lambda x:x[0], reverse=True)]
        batch_spks = [x for _, x in sorted(
            zip(batch_lengths, batch_spks), key=lambda x:x[0], reverse=True)]
        batch_texts = [x for _, x in sorted(
            zip(batch_lengths, batch_texts), key=lambda x:x[0], reverse=True)]
        batch_word_ids = [x for _, x in sorted(
            zip(batch_lengths, batch_word_ids), key=lambda x:x[0], reverse=True)]

        batch_lengths = sorted(batch_lengths, reverse=True)

        return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
                'lengths': torch.tensor(batch_lengths, dtype=torch.long),
                'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                'targets': batch_targets,
                'names': batch_names,
                'spks': batch_spks,
                'word_ids': batch_word_ids,
                'texts': batch_texts}

    if 'feats' in list_of_samples[0].keys():
        batch_targets = [sample['target'] for sample in list_of_samples]
        batch_target_lengths = [sample['target_length']
                                for sample in list_of_samples]
        batch_names = [sample['name'] for sample in list_of_samples]
        batch_spks = [sample['spk'] for sample in list_of_samples]
        batch_texts = [sample['text'] for sample in list_of_samples]
        batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

        batch_feats = [sample['feats'] for sample in list_of_samples]
        batch_feats_lens = [sample['feats_len'] for sample in list_of_samples]

        # Sorted
        batch_feats = [x for _, x in sorted(
            zip(batch_feats_lens, batch_feats), key=lambda x:x[0], reverse=True)]
        batch_targets = [x for _, x in sorted(
            zip(batch_feats_lens, batch_targets), key=lambda x:x[0], reverse=True)]
        batch_target_lengths = [x for _, x in sorted(
            zip(batch_feats_lens, batch_target_lengths), key=lambda x:x[0], reverse=True)]

        batch_names = [x for _, x in sorted(
            zip(batch_feats_lens, batch_names), key=lambda x:x[0], reverse=True)]
        batch_spks = [x for _, x in sorted(
            zip(batch_feats_lens, batch_spks), key=lambda x:x[0], reverse=True)]
        batch_texts = [x for _, x in sorted(
            zip(batch_feats_lens, batch_texts), key=lambda x:x[0], reverse=True)]
        batch_word_ids = [x for _, x in sorted(
            zip(batch_feats_lens, batch_word_ids), key=lambda x:x[0], reverse=True)]

        batch_feats_lens= sorted(batch_feats_lens, reverse=True)

        return {'feats': torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True),
                'feats_lens': torch.tensor(batch_feats_lens, dtype=torch.long),
                'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                'targets': batch_targets,
                'names': batch_names,
                'spks': batch_spks,
                'word_ids': batch_word_ids,
                'texts': batch_texts}

