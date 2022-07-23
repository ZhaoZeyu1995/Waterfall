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

        self.wav_scp = os.path.join(self.data_dir, 'wav.scp')
        self.utt2wav = kaldiio.load_scp(self.wav_scp)

        self.uttids = read_keys(self.wav_scp)
        self.utt2spk = read_dict(os.path.join(self.data_dir, 'utt2spk'))
        self.utt2text = read_dict(os.path.join(self.data_dir, 'text'))

        self.transforms = transforms

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        text = self.utt2text[uttid]
        target = tokenise(text, self.lang.token2idx)
        words = text.split(' ')
        word_ids = []
        for word in words:
            if word in self.lang.word2idx.keys():
                word_ids.append(self.lang.word2idx[word])
            else:
                word_ids.append(self.lang.word2idx['<UNK>']) # by default all unknown word should be denoted by <UNK>
        rate, wav = self.utt2wav[uttid]
        wav = torch.tensor(wav, dtype=torch.float32)
        if rate != RATE:
            wav = torchaudio.functional.resample(wav, rate, RATE)

        wav = (wav - wav.mean()) / torch.sqrt(wav.var())  # Normalisation

        sample = {'wav': wav,
                  'length': len(wav),
                  'target_length': len(target),
                  'target': target,
                  'name': uttid,
                  'spk': spk,
                  'word_ids': word_ids,
                  'text': text}
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

    batch_wav = [sample['wav'] for sample in list_of_samples]
    batch_lengths = [sample['length'] for sample in list_of_samples]

    return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
            'targets': batch_targets,
            'names': batch_names,
            'spks': batch_spks,
            'texts': batch_texts}


def collate_fn_sorted(list_of_samples):
    # Collect data
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
