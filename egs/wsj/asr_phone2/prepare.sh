#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./env.sh || exit 1;
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

topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"
lm_suffixes="test_bd_tg test_bd_fg"


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
    for x in train_si284_sp test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        steps/compute_cmvn_stats.sh data/${x}
        utils/fix_data_dir.sh data/${x}
        dump_utt2spk.sh data/${x} exp/dump_utt2spk/${x} data/${x}/dump/delta${do_delta}
    done

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Prepare Dictionary for phonemes"

    local/prepare_phone_dict.sh

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone \
                    "<UNK>" data/local/dict_phone_tmp data/lang
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone_test \
                    "<UNK>" data/local/dict_phone_test_tmp data/lang_eval

    cp -r data/lang_eval data/lang_eval_bd
    local/wsj_format_data.sh --lang-suffix "_eval"
    local/wsj_format_local_lms.sh --lang-suffix "_eval"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Generating different topologies and token FSTs for training."
    for topo in $topos; do
        [ -d data/lang_${topo} ] && rm -rf data/lang_${topo}
        cp -r data/lang data/lang_${topo}
        prepare_${topo}.sh data/lang_${topo}
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generating different topologies and token FSTs for evaluation."
    for topo in $topos; do
        for suffix in $lm_suffixes; do
            prepare_graph.sh --topo $topo data/lang_eval_${suffix} data/local/lang_eval_${suffix}_${topo}_tmp
        done
    done
fi
