#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=4

topos="ctc mmictc 2state 2state_blk mmictc_blk"
lm_suffixes="test_tgsmall test_tgmed test_tglarge test_fglarge"

# data
data=$LOCAL_HOME/data/librispeech
lm_url=www.openslr.org/resources/11


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "stage 0: data preparation"
    # format the data as Kaldi data directories
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
        local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
    done

fi

## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step. The pre-built language models and the pronunciation lexicon, as
## well as some intermediate data(e.g. the normalized text used for LM training),
## are available for download at http://www.openslr.org/11/
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm

## Optional G2P training scripts.
## As the LM training scripts above, this script is intended primarily to
## document our G2P model creation process
#local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    echo "stage 1: dictionary preparation"
    # download the LM resources
    local/download_lm.sh $lm_url data/local/lm

    local/prepare_dict.sh --stage 3 --nj 10 --cmd "$train_cmd" \
        data/local/lm data/local/lm data/local/dict_nosp

    utils/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
        "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

    local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: combine data "
    utils/combine_data.sh data/train_clean_460 data/train_clean_100 data/train_clean_360
    utils/combine_data.sh data/train_960 data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh data/dev data/dev_other/ data/dev_clean
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Generating different topologies and token FSTs."
    for topo in $topos; do
        for suffix in $lm_suffixes; do
            prepare_graph.sh --type $topo data/lang_nosp_${suffix} data/local/lang_nosp_${suffix}_${topo}_tmp
        done
    done
fi
