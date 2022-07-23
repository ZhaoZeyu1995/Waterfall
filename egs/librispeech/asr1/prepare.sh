#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
nj=4

#n-gram lm related
ngram_lm=/group/corpora/large4/librispeech/11/4-gram.arpa

# data
datadir=$LOCAL_HOME/data/librispeech


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_100
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done


fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: speech_perturb and combine data "
    local/wav2vec/perturb_data_dir_speed.sh 0.9 data/train_clean_100 data/train_clean_100_0.9
    local/wav2vec/perturb_data_dir_speed.sh 1.1 data/train_clean_100 data/train_clean_100_1.1
    utils/combine_data.sh data/train_clean_100_sp data/train_clean_100 data/train_clean_100_0.9 data/train_clean_100_1.1
    utils/combine_data.sh data/dev data/dev_other/ data/dev_clean
fi

local_dict=data/local/dict
lang_dir=data/lang
decode_langdir=data/lang_fg
mkdir -p $local_dict $lang_dir $decode_langdir
phones=$local_dict/phones.txt
nlsyms=$local_dict/non_lang_syms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary"

    # Just to make sure that ${nlsyms} is not empty to continue this programme
    echo "make a non-linguistic symbol list"
    echo "test <somethingrandom>" | cat - data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a phone set (phones.txt)"
    echo "<unk> 1" > ${phones} 
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${phones}
    wc -l ${phones}

    prepare_dict.sh --arpa $ngram_lm --local_dict $local_dict
    utils/prepare_lang.sh --position-dependent-phones false $local_dict "<UNK>" ${local_dict}_tmp $lang_dir

    prepare_ctc.sh $lang_dir
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: compute transition and transform matrix"
    compute_transition_and_transform_mat.sh data/$train_set $lang_dir
    compute_transition_and_transform_mat.sh data/$train_dev $lang_dir
    compute_transition_and_transform_mat.sh data/train_clean_100_sp $lang_dir
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: prepare decoding graph"
    mkgraph.sh $lang_dir $ngram_lm data/local/lang_tmp $decode_langdir || exit 1;
fi
