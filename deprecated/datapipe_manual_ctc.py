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
Data pipeline for kaldi data dir for manual ctc (for manual forward backward.).
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
                 transforms=None):
        '''
        Read a Kaldi data dir, which should usually contain

        wav.scp
        utt2spk
        text
        spk2utt
        spk2gender
        transition_mat.scp # transition matrix used for manual forward-backward


        TODO: This is no need to process feats.scp and cmvn.scp as 
        currently we only train our models based on wav2vec 2.0

        '''
        self.data_dir = data_dir
        self.lang_dir = lang_dir
        self.lang = Lang(self.lang_dir)

        self.wav_scp = os.path.join(self.data_dir, 'wav.scp')
        self.utt2wav = kaldiio.load_scp(self.wav_scp)

        self.uttids = read_keys(self.wav_scp)
        self.utt2spk = read_dict(os.path.join(self.data_dir, 'utt2spk'))
        self.utt2text = read_dict(os.path.join(self.data_dir, 'text'))
        self.utt2trans = kaldiio.load_scp(
            os.path.join(self.data_dir, 'transition_mat.scp'))
        self.utt2transform = kaldiio.load_scp(
            os.path.join(self.data_dir, 'transform_mat.scp'))

        self.transforms = transforms

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        text = self.utt2text[uttid]
        target = tokenise(text, self.lang.token2idx)
        rate, wav = self.utt2wav[uttid]
        wav = torch.tensor(wav)

        trans = torch.tensor(self.utt2trans[uttid])
        trans_length = len(trans)  # len(trans) == trans.shape[0] == S
        transform = torch.tensor(self.utt2transform[uttid])
        transform_length = len(transform) # len(transform) == transform.shape[0] == S

        assert transform_length == trans_length

        if rate != RATE:
            wav = torchaudio.functional.resample(wav, rate, RATE)

        wav = wav / torch.max(torch.abs(wav))  # Normalisation

        sample = {'wav': wav,
                  'length': len(wav),
                  'target_length': len(target),
                  'target': torch.tensor(target, dtype=torch.long),
                  'trans': trans,
                  'trans_length': trans_length,
                  'transform': transform,
                  'name': uttid,
                  'spk': spk,
                  'text': text}
        if self.transforms:
            return self.transforms(sample)
        else:
            return sample


def collate_fn_sorted(list_of_samples):

    # Collect all data
    batch_targets = [sample['target'] for sample in list_of_samples]
    batch_target_lengths = [sample['target_length']
                            for sample in list_of_samples]
    batch_names = [sample['name'] for sample in list_of_samples]
    batch_spks = [sample['spk'] for sample in list_of_samples]
    batch_texts = [sample['text'] for sample in list_of_samples]

    batch_wav = [sample['wav'] for sample in list_of_samples]
    batch_lengths = [sample['length'] for sample in list_of_samples]

    batch_trans = [sample['trans'] for sample in list_of_samples]
    batch_trans_lengths = [sample['trans_length']
                           for sample in list_of_samples]
    batch_transform = [sample['transform'] for sample in list_of_samples]

    # Sort data (perhaps optional for still apply as some API may demand)
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

    batch_trans_lengths = [x for _, x in sorted(
        zip(batch_lengths, batch_trans_lengths), key=lambda x:x[0], reverse=True)]
    batch_trans = [x for _, x in sorted(
        zip(batch_lengths, batch_trans), key=lambda x:x[0], reverse=True)]
    batch_transform = [x for _, x in sorted(
        zip(batch_lengths, batch_transform), key=lambda x:x[0], reverse=True)]

    batch_lengths = sorted(batch_lengths, reverse=True)

    return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
            'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
            'trans': pad_trans(batch_trans, max(batch_trans_lengths)),
            'trans_lengths': torch.tensor(batch_trans_lengths),
            'transform': pad_transform(batch_transform, max(batch_trans_lengths)),
            'names': batch_names,
            'spks': batch_spks,
            'texts': batch_texts}


def collate_fn(list_of_samples):

    # Collect all data
    batch_targets = [sample['target'] for sample in list_of_samples]
    batch_target_lengths = [sample['target_length']
                            for sample in list_of_samples]
    batch_names = [sample['name'] for sample in list_of_samples]
    batch_spks = [sample['spk'] for sample in list_of_samples]
    batch_texts = [sample['text'] for sample in list_of_samples]

    batch_wav = [sample['wav'] for sample in list_of_samples]
    batch_lengths = [sample['length'] for sample in list_of_samples]

    batch_trans = [sample['trans'] for sample in list_of_samples]
    batch_trans_lengths = [sample['trans_length']
                           for sample in list_of_samples]
    batch_transform = [sample['transform'] for sample in list_of_samples]

    return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
            'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
            'trans': pad_trans(batch_trans, max(batch_trans_lengths)),
            'trans_lengths': torch.tensor(batch_trans_lengths),
            'transform': pad_transform(batch_transform, max(batch_trans_lengths)),
            'names': batch_names,
            'spks': batch_spks,
            'texts': batch_texts}


def pad_trans(batch_trans, max_length):
    '''
    Pad a list of transition_mat (np.ndarray) in batch_trans according to max_length
    input:
    batch_trans: list of np.ndarray with shape (S_i, S_i)
    max_length: int, S_m, the maximum of S_i

    return:
    batch_trans_padded: torch.Tensor with shape (B, S_m, S_m)
    '''

    B = len(batch_trans)
    batch_trans_padded = torch.full(
        (B, max_length, max_length), torch.finfo(torch.float32).min)
    for idx, trans in enumerate(batch_trans):
        batch_trans_padded[idx, :trans.shape[0], :trans.shape[0]] = trans
    return batch_trans_padded


def pad_transform(batch_trans, max_length):
    '''
    Pad a list of transition_mat (np.ndarray) in batch_trans according to max_length
    input:
    batch_trans: list of np.ndarray with shape (S_i, C), where S_i is the length of the auxiliary sequence and C is the number of tokens.
    max_length: int, S_m, the maximum of S_i

    return:
    batch_trans_padded: torch.Tensor with shape (B, S_m, C)
    '''

    B = len(batch_trans)
    C = batch_trans[0].shape[-1]
    batch_trans_padded = torch.full(
        (B, max_length, C), torch.finfo(torch.float32).min)
    for idx, trans in enumerate(batch_trans):
        batch_trans_padded[idx, :trans.shape[0], :] = trans
    return batch_trans_padded
