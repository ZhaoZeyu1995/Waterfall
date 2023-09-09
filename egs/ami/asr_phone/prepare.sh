#!/bin/bash

# Copyright 2023 University of Edinburgh (Zeyu Zhao)
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
lm_suffixes="tg tgpr"

# data
data=$LOCAL_HOME/data/librispeech

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm

# exp tag
tag="" # tag for managing experiments.


. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

base_mic=${mic//[0-9]/} # sdm, ihm or mdm
nmics=${mic//[a-z]/} # e.g. 8 for mdm8.

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=db/ami # Default,

train_set=${mic}_train
train_dev=${mic}_dev
recog_set="${mic}_dev ${mic}_eval"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    if [ -d ${AMI_DIR} ] && ! touch ${AMI_DIR}/.foo 2>/dev/null; then
        echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
        echo " ... Assuming the data does not need to be downloaded.  Please use --stage 0 or more."
        exit 1
    fi
    if [ -e data/local/downloads/wget_${mic}.sh ]; then
        echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
        exit 1
    fi
    local/ami_download.sh ${mic} ${AMI_DIR}
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # common data prep
    if [ ! -d data/local/downloads/annotations ]; then
        local/ami_text_prep.sh data/local/downloads
    fi

    # beamforming
    if [ "$base_mic" == "mdm" ]; then
        PROCESSED_AMI_DIR=${PWD}/beamformed
        if [ -z ${BEAMFORMIT} ]; then
            export BEAMFORMIT=${KALDI_ROOT}/tools/BeamformIt
        fi
        export PATH=${PATH}:${BEAMFORMIT}
        ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/kaldi/tools; extras/install_beamformit.sh; cd -;'" && exit 1
        local/ami_beamform.sh --cmd "$train_cmd" --nj 20 ${nmics} ${AMI_DIR} ${PROCESSED_AMI_DIR}
    else
        PROCESSED_AMI_DIR=${AMI_DIR}
    fi
    local/ami_${base_mic}_data_prep.sh ${PROCESSED_AMI_DIR} ${mic}
    # data augmentation
    utils/perturb_data_dir_speed.sh 0.9 data/${mic}/train_orig data/${mic}_tmp1
    utils/perturb_data_dir_speed.sh 1.0 data/${mic}/train_orig data/${mic}_tmp2
    utils/perturb_data_dir_speed.sh 1.1 data/${mic}/train_orig data/${mic}_tmp3
    rm -r data/${mic}/train_orig
    utils/combine_data.sh --extra-files utt2uniq data/${mic}/train_orig data/${mic}_tmp1 data/${mic}_tmp2 data/${mic}_tmp3

    local/ami_${base_mic}_scoring_data_prep.sh ${PROCESSED_AMI_DIR} ${mic} dev
    local/ami_${base_mic}_scoring_data_prep.sh ${PROCESSED_AMI_DIR} ${mic} eval
    for dset in train dev eval; do
        # changed the original AMI data structure in the Kaldi recipe to the following
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 data/${mic}/${dset}_orig data/${mic}_${dset}
    done

    # get duration and num_frames 
    for dset in train dev eval; do
        utils/data/get_utt2num_frames.sh data/${mic}_${dset}
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
    for x in train dev eval; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${mic}_${x} exp/make_fbank/${mic}_${x} ${fbankdir} || exit 1
        steps/compute_cmvn_stats.sh data/${mic}_${x} || exit 1
        utils/fix_data_dir.sh data/${mic}_${x} || exit 1
    done

    for x in train dev eval; do
        feat_tr_dir=data/${mic}_${x}/dump/delta${do_delta}; mkdir -p ${feat_tr_dir}
        dump_utt2spk.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${mic}_${x} exp/dump_feats/${mic}_${x} ${feat_tr_dir} || exit 1
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

    echo "stage 3: dictionary preparation"

    local/prepare_phone_dict.sh || exit 1

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone \
        "<UNK>" data/local/lang_phone_tmp data/lang || exit 1
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_phone_test \
        "<UNK>" data/local/lang_phone_test_tmp data/lang_eval || exit 1
    utils/format_lm.sh data/lang_eval inputs/ami_fsh.o3g.kn.gz data/local/dict_phone_test/lexicon.txt data/lang_eval_tg || exit 1
    utils/format_lm.sh data/lang_eval inputs/ami_fsh.o3g.kn.pr1-7.gz data/local/dict_phone_test/lexicon.txt data/lang_eval_tgpr || exit 1
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
