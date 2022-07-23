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
ngram_lm=data/local/nist_lm/lm_tg.arpa

# data
wsj0=/group/corporapublic/wsj/wsj0
wsj1=/group/corporapublic/wsj/wsj1
corpus=/group/corporapublic/wsj


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
    #local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/cstr_wsj_data_prep.sh $corpus || exit 1;
    local/wsj_format_data.sh || exit 1;
    if [ -f data/local/nist_lm/lm_tg.arpa.gz ]; then
        gunzip -c data/local/nist_lm/lm_tg.arpa.gz > data/local/nist_lm/lm_tg.arpa || exit 1;
        ngram_lm=data/local/nist_lm/lm_tg.arpa
    else
        echo "Cannot find the language model $ngram_lm" && exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: speed perturbation and combine data"
    local/wav2vec/perturb_data_dir_speed.sh 0.9 data/$train_set data/${train_set}_0.9 || exit 1;
    local/wav2vec/perturb_data_dir_speed.sh 1.1 data/$train_set data/${train_set}_1.1 || exit 1;
    utils/combine_data.sh  data/${train_set}_sp data/${train_set} data/${train_set}_0.9 data/${train_set}_1.1 || exit 1;
fi

local_dict=data/local/dict
lang_dir=data/lang
decode_langdir=data/lang_tg
mkdir -p $local_dict $lang_dir $decode_langdir
phones=$local_dict/phones.txt
nlsyms=$local_dict/non_lang_syms.txt

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary"

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms} || exit 1;
    cat ${nlsyms}

    echo "make a phone set (phones.txt)"
    echo "<unk> 1" > ${phones} 
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${phones} || exit 1;
    wc -l ${phones}

    local/prepare_dict.sh --arpa $ngram_lm --local_dict $local_dict || exit 1;
    utils/prepare_lang.sh --position-dependent-phones false $local_dict "<UNK>" ${local_dict}_tmp $lang_dir || exit 1;

    cp -r $lang_dir ${lang_dir}_2state
    cp -r $lang_dir ${lang_dir}_mmictc

    prepare_ctc.sh $lang_dir || exit 1;
    prepare_2state.sh ${lang_dir}_2state || exit 1;
    prepare_mmictc.sh ${lang_dir}_mmictc || exit 1;
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: compute transition and transform matrix"
    compute_transition_and_transform_mat.sh data/$train_set $lang_dir || exit 1;
    compute_transition_and_transform_mat.sh data/${train_set}_sp $lang_dir || exit 1;
    compute_transition_and_transform_mat.sh data/$train_dev $lang_dir || exit 1;
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: preparing decoding graph"
    mkgraph.sh $lang_dir $ngram_lm data/local/lang_tmp $decode_langdir || exit 1;
    mkgraph.sh ${lang_dir}_2state $ngram_lm data/local/lang_2state_tmp ${decode_langdir}_2state || exit 1;
    mkgraph.sh ${lang_dir}_mmictc $ngram_lm data/local/lang_mmictc_tmp ${decode_langdir}_mmictc || exit 1;
fi
