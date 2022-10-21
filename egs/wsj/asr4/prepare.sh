#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=10
do_delta=false

# data
wsj0=/group/corporapublic/wsj/wsj0
wsj1=/group/corporapublic/wsj/wsj1
corpus=/group/corporapublic/wsj

#topos="ctc mmictc 2state 2state_blk mmictc_blk"
topos="ctc 2state_blk mmictc_blk"
#lm_suffixes="test_bg test_bg_5k test_tg test_tg_5k test_tgpr test_tgpr_5k test_bd_fg test_bd_fgpr test_bd_tg test_bd_tgpr"
lm_suffixes="test_bg test_tg test_bd_fg test_bd_tg"

nbpe=500
bpemode=unigram


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

base_train_set=train_si284
train_set=train_si284_sp
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/cstr_wsj_data_prep.sh $corpus
    local/wsj_prepare_dict.sh --dict-suffix "_nosp"
    utils/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp
    local/wsj_format_data.sh --lang-suffix "_nosp"
    echo "Done formatting the data."

    local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" $corpus/wsj1/doc/
    utils/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp_larger \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger \
                        data/lang_nosp_bd
    local/wsj_train_lms.sh --dict-suffix "_nosp"
    local/wsj_format_local_lms.sh --lang-suffix "_nosp"
    echo "Done extending the dictionary and formatting LMs."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: speed perturbation and combine data"
    local/wav2vec/perturb_data_dir_speed.sh 0.9 data/${base_train_set} data/${base_train_set}_0.9 || exit 1;
    local/wav2vec/perturb_data_dir_speed.sh 1.1 data/$base_train_set data/${base_train_set}_1.1 || exit 1;
    utils/combine_data.sh  data/${train_set} data/${base_train_set} data/${base_train_set}_0.9 data/${base_train_set}_1.1 || exit 1;
fi

feat_tr_dir=data/${train_set}/dump/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=data/${train_dev}/dump/delta${do_delta}; mkdir -p ${feat_dt_dir}
fbankdir=fbank

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature extraction and dump"
    for x in train_si284 train_si284_0.9 train_si284_1.1 test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh  data/${train_set} data/${base_train_set} data/${base_train_set}_0.9 data/${base_train_set}_1.1 || exit 1;

    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=data/${rtask}/dump/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

    echo "stage 3: BPE dictionary preparation"
    # download the LM resources
    #local/download_lm.sh $lm_url data/local/lm

    mkdir -p data/local/dict_bpe_${nbpe}

    vocab_file=data/lang_nosp_bd/words.txt
    cat data/local/dict_nosp/cmudict/cmudict-0.7b | grep -v ";;;" | cut -d " " -f 1 > $vocab_file

    local/prepare_bpe_dict.sh --nbpe ${nbpe} --bpemode ${bpemode} data/local/dict_bpe_${nbpe} data/train_si284 $vocab_file

    utils/prepare_lang.sh --position-dependent-phones false data/local/dict_bpe_${nbpe} \
        "<UNK>" data/local/lang_tmp_bpe_${nbpe} data/lang_bpe_${nbpe}


    local/wsj_format_data.sh --lang-suffix "_bpe_${nbpe}"
    cp -r data/lang_bpe_${nbpe} data/lang_bpe_${nbpe}_bd
    local/wsj_format_local_lms.sh --lang-suffix "_bpe_${nbpe}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Generating different topologies and token FSTs."
    for topo in $topos; do
        for suffix in $lm_suffixes; do
            prepare_graph.sh --type $topo data/lang_bpe_${nbpe}_${suffix} data/local/lang_bpe_${nbpe}_${suffix}_${topo}_tmp
            mkdir -p data/lang_bpe_${nbpe}_${suffix}_${topo}/decode || exit 1;
            k2todecode_tokens.py data/lang_bpe_${nbpe}_${suffix}_${topo}/k2/tokens.txt > data/lang_bpe_${nbpe}_${suffix}_${topo}/decode/tokens.txt || exit 1;
        done
    done
fi
