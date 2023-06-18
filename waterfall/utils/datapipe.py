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
import k2
from waterfall import graph


'''
Data pipeline for kaldi data dir. 
'''

RATE = 16000  # Sample Rate for WAV2VEC2


def read_list(file):
    '''
    Read a list file, e.g, phones.txt, words.txt, tokens.txt, containing one single item per line
    Return a list of all elements
    '''
    with open(file) as f:
        items = []
        for line in f:
            items.append(line.strip().split()[0])
    return items


def read_dict(file, mapping=None):
    '''
    Read a dict file, e.g, wav.scp, text, with the pattern <key> <value>
    Return a dict[<key>] = <value>

    By default, <value> is a string, but can be transformed by mapping (None, by default)

    mapping: a function applied to transform the values 
    '''
    a = dict()
    with open(file) as f:
        for line in f:
            lc = line.strip().split()
            if mapping:
                assert len(lc) == 2, 'Expect two elements per line but got %d' % (
                    len(lc))
                a[lc[0]] = mapping(lc[1])
            else:
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
            keys.append(utt)
    return keys


class Lang(object):

    """This class is used to read a lang dir"""

    def __init__(self,
                 lang_dir,
                 load_topo=False,
                 load_lexicon=False):
        """
        It should contain the following files usually

        dir phones, created by Kaldi
        L.fst, created by Kaldi
        L_disambig.fst, created by Kaldi
        oov.int and oov.txt, created by Kaldi
        phones.txt, created by Kaldi
        tokens.txt, created by this repo, the output tokens of the neural network model
        topo, created by Kaldi
        words.txt, created by Kaldi

        Here are some some files introduced in this project.

        T.fst, created by this repo, token FST, for decoding
        disambig.int, the indexes of the disambig symbols in tokens_disambig.txt
        tokens_disambig.txt, for constructing the decoding graph only

        Besides, it may also contain a directory k2/ which has the following files for training only
        phones.txt, without disambig symbols
        tokens.txt, no <eps> nor disambig symbols, which is exactly reflecting the outputs of the neural network
        T.fst, for training only as it may not be deterministic

        args:
        lang_dir: str, the language directory
        load_topo, bool, False by default, whether or not load $lang_dir/k2/T.fst
        load_lexicon: bool, False by default, whether or not load lang_dir/k2/L_inv.pt. Generate one from lang_dir/L.fst if there is not.

        """
        self._lang_dir = lang_dir
        # with <eps> and disambig
        self.words = read_list(os.path.join(lang_dir, 'words.txt'))
        # with <eps> and disambig
        self.phones = read_list(os.path.join(lang_dir, 'phones.txt'))
        # no <eps> no disambig, to match the NN outputs
        self.tokens = read_list(os.path.join(lang_dir, 'k2', 'tokens.txt'))

        self.num_nn_output = len(self.tokens)

        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx2word = {idx: word for idx, word in enumerate(self.words)}

        self.phone2idx = {phone: idx for idx, phone in enumerate(self.phones)}
        self.idx2phone = {idx: phone for idx, phone in enumerate(self.phones)}

        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.load_align_lexicon()

        if load_topo:
            self.load_topo()

        if load_lexicon:
            self.load_lexicon()

    def load_align_lexicon(self):
        '''
        This function load the information from
        $lang_dir/phones/align_lexicon.int,
        which is a dict (int->[int]) mapping each word to its phone sequence,
        where the index is given by self.phone2idx and self.word2idx.
        '''

        self.lexicon = dict()
        with open(os.path.join(self._lang_dir, 'phones', 'align_lexicon.int')) as f:
            for line in f:
                lc = line.strip().split()
                wid = int(lc[0])
                pids = list(map(int, lc[2:]))
                self.lexicon[wid] = pids

    def load_topo_bak(self):
        '''
        For some reason, we may need to keep this for future development.

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
        self.topo_den = k2.remove_epsilon(
            k2.Fsa.from_openfst(openfst_txt, acceptor=False))
        print('Done!')

    def load_topo(self):
        '''
        Load $lang_dir/k2/T.fst 
        At the same time, we need to project the input labels to output labels,
        as the input labels represent tokens in tokens.txt and output labels represent phones in phones.txt.
        '''
        print(f'Loading and processing topo from {self._lang_dir}/k2/T.fst')
        cmd = (
            f"""fstprint {self._lang_dir}/k2/T.fst | """
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )
        openfst_txt = os.popen(cmd).read()
        self.topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        print('Done!')

    def load_lexicon(self):
        '''
        Load lang_dir/k2/L_inv.pt and generate one if there is not one already.
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
        '''
        Note!
        This part cannot work well as the composition of T and L can be too large.
        '''

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

    def get_pids_list_from_wids_list(self, wids_list):
        '''
        word_ids_list: [[int]]

        return:
        pids_list: [[int]]
        '''
        pids_list = []
        for wids in wids_list:
            pids = []
            for wid in wids:
                pids.extend(self.lexicon[wid])
            pids_list.append(pids)
        return pids_list

    def compile_training_graph_bak(self, word_ids_list, device):
        '''
        Note! For some reason, we keep this for future debugging and development!

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

        training_graph = k2.remove_epsilon(k2.compose(
            self.topo, trans_fsa_with_self_loop, treat_epsilons_specially=False))

        return training_graph


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 lang_dir,
                 ratio_th=None,
                 ctc_target=False,
                 load_wav=False,
                 load_feats=False,
                 do_delta=False,
                 transforms=None):
        '''
        Read a Kaldi data dir, which should usually contain

        wav.scp
        utt2spk
        text
        spk2utt
        utt2spk
        utt2dur

        args:
        data_dir: str, the data directory
        lang_dir: str, the language directory
        ctc_target: bool, whether or not prepare ctc targets which is stored with the key 'target_ctc', by default False
        load_wav: bool, whether or not load wav from wav.scp, by default False
        load_feats: bool, whether or not load feats from $data_dir/dump/delta{true/flase}/feats.scp, by default False
        do_delta: bool, whether or not load the delta feats , by default False
        transforms: func, a function that transforms samples, e.g, SpecAugment
        '''
        self.data_dir = data_dir
        self.lang_dir = lang_dir
        self.ratio_th = ratio_th
        self.lang = Lang(self.lang_dir)

        self.wav_scp = os.path.join(self.data_dir, 'wav.scp')
        if load_wav:
            self.utt2wav = kaldiio.load_scp(self.wav_scp)

        self.uttids = read_keys(self.wav_scp)
        self.utt2spk = read_dict(os.path.join(self.data_dir, 'utt2spk'))
        self.utt2text = read_dict(os.path.join(self.data_dir, 'text'))

        self.utt2dur = read_dict(os.path.join(
            self.data_dir, 'utt2dur'), mapping=float)
        self.utt2num_frames = read_dict(os.path.join(
            self.data_dir, 'utt2num_frames'), mapping=int)

        self.ctc_target = ctc_target
        self.load_wav = load_wav
        self.load_feats = load_feats

        if do_delta:
            self.dump_feats = os.path.join(
                self.data_dir, 'dump', 'deltatrue', 'feats.scp')
        else:
            self.dump_feats = os.path.join(
                self.data_dir, 'dump', 'deltafalse', 'feats.scp')

        if load_feats:
            self.utt2feats = kaldiio.load_scp(self.dump_feats)

        self.transforms = transforms

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, idx):
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        dur = self.utt2dur[uttid]
        num_frame = self.utt2num_frames[uttid]
        text = self.utt2text[uttid]
        words = text.split(' ')
        pids = []  # pids for ctc only
        word_ids = []
        for word in words:
            if word in self.lang.word2idx.keys():
                wid = self.lang.word2idx[word]
                word_ids.append(wid)
                # wid must be in self.lang.lexicon.keys(), which is guaranteed by how lang dir is generated
                pids.extend(self.lang.lexicon[wid])
            else:
                # by default all unknown words are denoted by <UNK>
                wid = self.lang.word2idx['<UNK>']
                word_ids.append(wid)
                pids.extend(self.lang.lexicon[wid])
        tids = []
        if self.ctc_target:
            for pid in pids:
                assert self.lang.idx2phone[pid] in self.lang.token2idx, 'Cannot find the token %s from the token list, please make sure you are using CTC topo' % (
                    self.lang.idx2phone[pid])
                tids.append(self.lang.token2idx[self.lang.idx2phone[pid]])

        # Check if the num_frame is enough, otherwise we just take the last item in the dataset to keep the same number of samples in one epoch
        # A typical value is 8.5 because a common experiment setting is a subsampling facotr of 4 and the 2-state topology.
        # This leads to some loss of data by approximately 4% of the training data in WSJ. with BPE 100.
        # However, we should definitely keep ratio_th as None during evaluation.
        if self.ratio_th:
            if int(num_frame / self.ratio_th) < len(pids):
                return self.__getitem__(idx-1)

        sample = {
            # Note this is for CTC and reference only but not for DWFST-based training
            'target_length': len(pids),
            'target': torch.tensor(pids, dtype=torch.int64),
            'target_ctc': torch.tensor(tids, dtype=torch.int64),
            # Note this is for CTC and reference only but not for DWFST-based training
            'name': uttid,
            'spk': spk,
            'dur': dur,
            'num_frame': num_frame,
            'word_ids': word_ids,
            'text': text
        }

        if self.load_wav:
            rate, wav = self.utt2wav[uttid]
            wav = torch.tensor(wav, dtype=torch.float32)
            if rate != RATE:
                wav = torchaudio.functional.resample(wav, rate, RATE)

            wav = (wav - wav.mean()) / torch.sqrt(wav.var())  # Normalisation
            sample['wav'] = wav
            sample['wav_len'] = len(wav)

        if self.load_feats:
            feats = torch.tensor(self.utt2feats[uttid], dtype=torch.float32)
            sample['feats'] = feats
            sample['feats_len'] = feats.shape[0]

        if self.transforms:
            return self.transforms(sample)
        else:
            return sample


def collate_fn(list_of_samples):
    batch_targets = [sample['target'] for sample in list_of_samples]
    batch_targets_ctc = [sample['target_ctc'] for sample in list_of_samples]
    batch_target_lengths = [sample['target_length']
                            for sample in list_of_samples]
    batch_names = [sample['name'] for sample in list_of_samples]
    batch_spks = [sample['spk'] for sample in list_of_samples]
    batch_durs = [sample['dur'] for sample in list_of_samples]
    batch_num_frames = [sample['num_frame'] for sample in list_of_samples]
    batch_texts = [sample['text'] for sample in list_of_samples]
    batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

    samples_collated = {'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                        'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
                        'targets_ctc': torch.nn.utils.rnn.pad_sequence(batch_targets_ctc, batch_first=True),
                        'names': batch_names,
                        'spks': batch_spks,
                        'durs': batch_durs,
                        'num_frames': batch_num_frames,
                        'texts': batch_texts}

    if 'wav' in list_of_samples[0].keys():
        batch_wav = [sample['wav'] for sample in list_of_samples]
        batch_wav_lengths = [sample['wav_len'] for sample in list_of_samples]
        samples_collated['wavs'] = torch.nn.utils.rnn.pad_sequence(
            batch_wav, batch_first=True)
        samples_collated['wav_lens'] = torch.tensor(
            batch_wav_lengths, dtype=torch.long)

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
        batch_targets_ctc = [sample['target_ctc']
                             for sample in list_of_samples]
        batch_target_lengths = [sample['target_length']
                                for sample in list_of_samples]
        batch_names = [sample['name'] for sample in list_of_samples]
        batch_spks = [sample['spk'] for sample in list_of_samples]
        batch_durs = [sample['dur'] for sample in list_of_samples]
        batch_num_frames = [sample['num_frame'] for sample in list_of_samples]
        batch_texts = [sample['text'] for sample in list_of_samples]
        batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

        batch_wav = [sample['wav'] for sample in list_of_samples]
        batch_lengths = [sample['wav_len'] for sample in list_of_samples]

        # Sorted
        batch_wav = [x for _, x in sorted(
            zip(batch_lengths, batch_wav), key=lambda x:x[0], reverse=True)]
        batch_targets = [x for _, x in sorted(
            zip(batch_lengths, batch_targets), key=lambda x:x[0], reverse=True)]
        batch_targets_ctc = [x for _, x in sorted(
            zip(batch_lengths, batch_targets_ctc), key=lambda x:x[0], reverse=True)]
        batch_target_lengths = [x for _, x in sorted(
            zip(batch_lengths, batch_target_lengths), key=lambda x:x[0], reverse=True)]

        batch_names = [x for _, x in sorted(
            zip(batch_lengths, batch_names), key=lambda x:x[0], reverse=True)]
        batch_spks = [x for _, x in sorted(
            zip(batch_lengths, batch_spks), key=lambda x:x[0], reverse=True)]
        batch_durs = [x for _, x in sorted(
            zip(batch_lengths, batch_durs), key=lambda x:x[0], reverse=True)]
        batch_num_frames = [x for _, x in sorted(
            zip(batch_lengths, batch_num_frames), key=lambda x:x[0], reverse=True)]
        batch_texts = [x for _, x in sorted(
            zip(batch_lengths, batch_texts), key=lambda x:x[0], reverse=True)]
        batch_word_ids = [x for _, x in sorted(
            zip(batch_lengths, batch_word_ids), key=lambda x:x[0], reverse=True)]

        batch_lengths = sorted(batch_lengths, reverse=True)

        return {'wavs': torch.nn.utils.rnn.pad_sequence(batch_wav, batch_first=True),
                'wav_lens': torch.tensor(batch_lengths, dtype=torch.long),
                'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
                'targets_ctc': torch.nn.utils.rnn.pad_sequence(batch_targets_ctc, batch_first=True),
                'names': batch_names,
                'spks': batch_spks,
                'durs': batch_durs,
                'num_frames': batch_num_frames,
                'word_ids': batch_word_ids,
                'texts': batch_texts}

    if 'feats' in list_of_samples[0].keys():
        batch_targets = [sample['target'] for sample in list_of_samples]
        batch_targets_ctc = [sample['target_ctc']
                             for sample in list_of_samples]
        batch_target_lengths = [sample['target_length']
                                for sample in list_of_samples]
        batch_names = [sample['name'] for sample in list_of_samples]
        batch_spks = [sample['spk'] for sample in list_of_samples]
        batch_durs = [sample['dur'] for sample in list_of_samples]
        batch_num_frames = [sample['num_frame'] for sample in list_of_samples]
        batch_texts = [sample['text'] for sample in list_of_samples]
        batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

        batch_feats = [sample['feats'] for sample in list_of_samples]
        batch_feats_lens = [sample['feats_len'] for sample in list_of_samples]

        # Sorted
        batch_feats = [x for _, x in sorted(
            zip(batch_feats_lens, batch_feats), key=lambda x:x[0], reverse=True)]
        batch_targets = [x for _, x in sorted(
            zip(batch_feats_lens, batch_targets), key=lambda x:x[0], reverse=True)]
        batch_targets_ctc = [x for _, x in sorted(
            zip(batch_feats_lens, batch_targets_ctc), key=lambda x:x[0], reverse=True)]
        batch_target_lengths = [x for _, x in sorted(
            zip(batch_feats_lens, batch_target_lengths), key=lambda x:x[0], reverse=True)]

        batch_names = [x for _, x in sorted(
            zip(batch_feats_lens, batch_names), key=lambda x:x[0], reverse=True)]
        batch_spks = [x for _, x in sorted(
            zip(batch_feats_lens, batch_spks), key=lambda x:x[0], reverse=True)]
        batch_durs = [x for _, x in sorted(
            zip(batch_feats_lens, batch_durs), key=lambda x:x[0], reverse=True)]
        batch_num_frames = [x for _, x in sorted(
            zip(batch_feats_lens, batch_num_frames), key=lambda x:x[0], reverse=True)]
        batch_texts = [x for _, x in sorted(
            zip(batch_feats_lens, batch_texts), key=lambda x:x[0], reverse=True)]
        batch_word_ids = [x for _, x in sorted(
            zip(batch_feats_lens, batch_word_ids), key=lambda x:x[0], reverse=True)]

        batch_feats_lens = sorted(batch_feats_lens, reverse=True)

        return {'feats': torch.nn.utils.rnn.pad_sequence(batch_feats, batch_first=True),
                'feats_lens': torch.tensor(batch_feats_lens, dtype=torch.long),
                'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.long),
                'targets': torch.nn.utils.rnn.pad_sequence(batch_targets, batch_first=True),
                'targets_ctc': torch.nn.utils.rnn.pad_sequence(batch_targets_ctc, batch_first=True),
                'names': batch_names,
                'spks': batch_spks,
                'word_ids': batch_word_ids,
                'texts': batch_texts}
