#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1
. ./env.sh || exit 1
. ./cmd.sh || exit 1

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=10

do_delta=false

topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"
lm_suffixes="fgsmall fgbig"

# data
data=$LOCAL_HOME/data/librispeech
lm_url=www.openslr.org/resources/11


. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
recog_set="dev test"


if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    local/download_data.sh || exit 1
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "stage 0: data preparation"
    # format the data as Kaldi data directories
    local/prepare_data.sh || exit 1
    # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
    # lets us use more jobs for decoding etc.
    # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
    #  more than our normal 30 jobs.]
    for dset in dev test train; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset} || exit 1
    done

    for x in train dev test; do
        utils/data/get_utt2num_frames.sh data/$x || exit 1
    done
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: combine data "
    echo "Pass"
fi



feat_tr_dir=data/${train_set}/dump/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=data/${train_dev}/dump/delta${do_delta}; mkdir -p ${feat_dt_dir}
fbankdir=fbank

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: compute features "
    # The feature extraction can be skipped if we only train wav2vec 2.0 model
    for x in train dev test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir} || exit 1
        steps/compute_cmvn_stats.sh data/${x} || exit 1
        utils/fix_data_dir.sh data/${x} || exit 1
    done

    for x in train dev test; do
        feat_tr_dir=data/${x}/dump/delta${do_delta}; mkdir -p ${feat_tr_dir}
        dump_utt2spk.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${x} exp/dump_feats/${x} ${feat_tr_dir} || exit 1
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

    echo "stage 3: dictionary preparation"

    local/prepare_phone_dict.sh || exit 1

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone \
        "<UNK>" data/local/lang_phone_tmp data/lang || exit 1
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone_test \
        "<UNK>" data/local/lang_phone_test_tmp data/lang_eval || exit 1

    local/ted_download_lm.sh || exit 1
    local/format_lms.sh data/lang_eval || exit 1
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
