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
import k2
from waterfall import graph


'''
Data pipeline for kaldi data dir.
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


class Lang(object):

    """This class is used to read a lang dir"""

    def __init__(self, lang_dir,
                 token_type='phones',
                 load_topo=False,
                 load_lexicon=False,
                 load_den_graph=False):
        """

        :lang: the lang dir in kaldi (utils/prepare_lang.sh) with some other files created by this repo

        It should contain the following files usually

        dir phones, created by Kaldi
        disambig.int, created by this repo, the indexes of the disambig symbols in tokens_disambig.txt
        L.fst, created by Kaldi
        L_disambig.fst, created by Kaldi
        oov.int and oov.txt, created by Kaldi
        phones.txt, created by Kaldi
        T.fst, created by this repo, token FST
        tokens.txt, created by this repo, the output tokens of the neural network model
        tokens_disambig.txt, created by this repo, for constructing the decoding graph only
        topo, created by Kaldi
        words.txt, created by Kaldi

        args:
        lang_dir: str, the lang dir
        load_topo, bool, False by default, whether or not load lang_dir/k2/T.fst
        token_type, str, ['tokens', 'phones'] if 'tokens', load lang_dir/tokens.txt, if 'phones' load lang_dir/phones.txt
            but remove <eps> and disambig symbols, by default tokens.
        load_lexicon: bool, False by default, whether or not load lang_dir/k2/L_inv.pt. Generate one from lang_dir/L.fst if there is not.

        """
        self._lang_dir = lang_dir
        self.phones = read_list(os.path.join(lang_dir, 'phones.txt'))
        self.tokens = read_list(os.path.join(lang_dir, 'tokens.txt'))
        self.words = read_list(os.path.join(lang_dir, 'words.txt'))

        nn_output_tokens = [
            item for item in self.tokens if item != '<eps>' and not item.startswith('#')]
        self.num_nn_output = len(nn_output_tokens)

        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx2word = {idx: word for idx, word in enumerate(self.words)}

        # For this two dicts, we need to get rid of <eps> (which is used for openfst)
        # assume that <blk> is the first token in tokens.txt
        # This is usually for ctc training, as the token set and phone set are almost the same, except the <blk>
        # in the token set, in CTC.
        if token_type == 'tokens':
            tokens = self.tokens[1:]
            self.token2idx = {token: idx for idx, token in enumerate(tokens)}
            self.idx2token = {idx: token for idx, token in enumerate(tokens)}
        # For phones, remove <eps> and disambig symbols and there is no <blk> in phones
        # This is generally applied for k2 graph training.
        elif token_type == 'phones':
            tokens = [token for token in self.phones if token !=
                      '<eps>' and not token.startswith('#')]
            self.token2idx = {token: idx for idx,
                              token in enumerate(tokens, start=1)}
            self.idx2token = {idx: token for idx,
                              token in enumerate(tokens, start=1)}

        if load_topo:
            self.load_topo()

        if load_lexicon:
            self.load_lexicon()

        if load_den_graph:
            pass
            # self.compile_denominator_graph()

    def load_topo(self):
        '''
        Load T.fst in the lang_dir and transform it to k2 format for training or decoding
        At the same time, we need to project the input labels to output labels,
        as the input labels represent tokens in tokens.txt and output labels represent phones in phones.txt.
        '''
        print(f'Loading topo from {self._lang_dir}/T.fst')
        cmd = (
            f"""fstprint {self._lang_dir}/T.fst | """
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )
        openfst_txt = os.popen(cmd).read()
        self.topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        print('Done!')

    def load_topo_bak(self):
        '''
        Load T.fst in the lang_dir and transform it to k2 format for training or decoding
        At the same time, we need to project the input labels to output labels,
        as the input labels represent tokens in tokens.txt and output labels represent phones in phones.txt.
        '''
        print(f'Loading topo from {self._lang_dir}/k2/T.fst')
        cmd = (
            f"""fstprint {self._lang_dir}/k2/T.fst | """
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )
        openfst_txt = os.popen(cmd).read()
        self.topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        print('Done!')

    def load_lexicon(self):
        '''
        Load lang_dir/k2/L_inv.pt and generate one if there is not.
        '''
        L_inv_fst = os.path.join(self._lang_dir, 'k2', 'L_inv.pt')
        if os.path.exists(L_inv_fst):
            print(f'Loading L_inv from {self._lang_dir}/k2/L_inv.pt')
            self.L_inv = k2.arc_sort(k2.Fsa.from_dict(torch.load(L_inv_fst)))
            self.L = k2.arc_sort(self.L_inv.invert())
        else:
            print(
                f'Loading {self._lang_dir}/L.fst and transforming it into {self._lang_dir}/k2/L_inv.pt ')

            cmd = (
                f"""fstprint {self._lang_dir}/L.fst | """
                f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
            )
            openfst_txt = os.popen(cmd).read()
            self.L = k2.arc_sort(k2.Fsa.from_openfst(
                openfst_txt, acceptor=False))
            self.L_inv = k2.arc_sort(self.L.invert())
            torch.save(self.L_inv.as_dict(), L_inv_fst)

    def compile_denominator_graph(self):

        if not os.path.exists(f"{self._lang_dir}/k2/TL.fst"):
            print(
                f'Composing {self._lang_dir}/k2/T.fst and {self._lang_dir}/L.fst for the denominator')
            cmd = (
                f"""fstarcsort --sort_type=ilabel {self._lang_dir}/L.fst |"""
                f"""fsttablecompose {self._lang_dir}/k2/T.fst - |"""
                f"""fstrmepslocal |"""
                f"""fstminimizeencoded |"""
                f"""fstarcsort --sort_type=ilabel > {self._lang_dir}/k2/TL.fst"""
            )
            stream = os.popen(cmd)
            print(stream.read())
        print(f'Loading {self._lang_dir}/k2/TL.fst')
        cmd = (
            f"""fstprint {self._lang_dir}/k2/TL.fst |"""
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )

        self.den_graph = k2.arc_sort(k2.Fsa.from_openfst(
            os.popen(cmd).read(), acceptor=False))

    def compile_training_graph(self, word_ids_list, device):
        '''
        word_ids_list: a list of lists of words_ids
        device: str, the computing device, usually should be log_probs.device where log_probs is the NN outputs

        return:
        the training fst according to self.topo and self.L_inv
        '''
        self.topo = self.topo.to(device)
        self.L_inv = self.L_inv.to(device)

        word_fsa = k2.linear_fsa(word_ids_list, device=device)
        word_fsa_with_self_loop = k2.add_epsilon_self_loops(word_fsa)
        fsa = k2.intersect(self.L_inv, word_fsa_with_self_loop,
                           treat_epsilons_specially=False)

        trans_fsa = k2.arc_sort(fsa.invert())  # trans_fsa: phones -> words
        trans_fsa_with_self_loop = k2.arc_sort(
            k2.remove_epsilon_and_add_self_loops(trans_fsa))

        training_graph = k2.compose(
            self.topo, trans_fsa_with_self_loop, treat_epsilons_specially=False)

        return training_graph


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

        self.transforms = transforms

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        text = self.utt2text[uttid]
        target = tokenise(text, self.lang.token2idx)
        rate, wav = self.utt2wav[uttid]
        wav = torch.tensor(wav, dtype=torch.float32)
        if rate != RATE:
            wav = torchaudio.functional.resample(wav, rate, RATE)

        wav = (wav - wav.mean()) / torch.sqrt(wav.var())  # Normalisation

        sample = {'wav': wav,
                  'length': len(wav),
                  'target_length': len(target),
                  'target': torch.tensor(target, dtype=torch.long),
                  'name': uttid,
                  'spk': spk,
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

    batch_wav = [sample['wav'] for sample in list_of_samples]
    batch_lengths = [sample['length'] for sample in list_of_samples]

    return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
            'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
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

    batch_lengths = sorted(batch_lengths, reverse=True)

    return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
            'lengths': torch.tensor(batch_lengths, dtype=torch.long),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
            'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
            'names': batch_names,
            'spks': batch_spks,
            'texts': batch_texts}
