#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=4

# data
wsj0=/group/corporapublic/wsj/wsj0
wsj1=/group/corporapublic/wsj/wsj1
corpus=/group/corporapublic/wsj

topos="ctc mmictc 2state 2state_blk mmictc_blk"


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
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
    echo "Done exteding the dictionary and formatting LMs."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: speed perturbation and combine data"
    local/wav2vec/perturb_data_dir_speed.sh 0.9 data/$train_set data/${train_set}_0.9 || exit 1;
    local/wav2vec/perturb_data_dir_speed.sh 1.1 data/$train_set data/${train_set}_1.1 || exit 1;
    utils/combine_data.sh  data/${train_set}_sp data/${train_set} data/${train_set}_0.9 data/${train_set}_1.1 || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Generating different topologies and token FSTs."
    lm_suffixes="test_bg test_bg_5k test_tg test_tg_5k test_tgpr test_tgpr_5k test_bd_fg test_bd_fgpr test_bd_tg test_bd_tgpr"
    for topo in $topos; do
        for suffix in $lm_suffixes; do
            prepare_graph.sh --type $topo data/lang_nosp_${suffix} data/local/lang_nosp_${suffix}_${topo}_tmp
        done
    done
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: preparing decoding graph"
    mkgraph.sh $lang_dir $ngram_lm data/local/lang_tmp $decode_langdir || exit 1;
    mkgraph.sh ${lang_dir}_2state $ngram_lm data/local/lang_2state_tmp ${decode_langdir}_2state || exit 1;
    mkgraph.sh ${lang_dir}_mmictc $ngram_lm data/local/lang_mmictc_tmp ${decode_langdir}_mmictc || exit 1;
fi
