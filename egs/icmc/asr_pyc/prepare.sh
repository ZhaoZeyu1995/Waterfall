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

topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"
lm_suffixes="tg tgw"


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fbankdir=fbank
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 2: feature extraction and dump"
    for x in train_near train_aec_iva train_aec_iva_near dev_near dev_aec_iva; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        steps/compute_cmvn_stats.sh data/${x}
        utils/fix_data_dir.sh data/${x}
        dump_utt2spk.sh --nj ${nj} data/${x} exp/dump_utt2spk/${x} data/${x}/dump/delta${do_delta}
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Get Duration"
    utils/data/get_utt2num_frames.sh data/train_aec_iva_near
    utils/data/get_utt2num_frames.sh data/dev_aec_iva
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Prepare Dictionary for phonemes"

    local/prepare_phone_dict.sh

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone \
                    "<UNK>" data/local/dict_phone_tmp data/lang
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone_test \
                    "<UNK>" data/local/dict_phone_test_tmp data/lang_eval
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
