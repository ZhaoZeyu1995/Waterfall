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

topos="ctc mmictc 2state 2state_blk mmictc_blk"
lm_suffixes="test_tg test_bd_tg test_bd_fg"
nbpe=1000


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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Prepare Dictionary for BPE"

    bpe_local_dict=data/local/lang_bpe
    nlsyms=$bpe_local_dict/nlsyms
    if [ -d $bpe_local_dict ]; then
        rm -rf $bpe_local_dict
    fi
    mkdir -p $bpe_local_dict

    (cat data/train_si284/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq; cat data/lang_nosp_bd/words.txt | awk '{print $1}') \
        | sort | uniq | grep -v -E '<eps>|<s>|</s>|#0|^#' > $bpe_local_dict/words.txt  || exit 1;
    (echo "'APOSTROPHE"; echo "'END-INNER-QUOTE"; echo "'END-QUOTE"; echo "'INNER-QUOTE"; echo "'QUOTE"; echo "'SINGLE-QUOTE") > ${nlsyms}_tmp || exit 1;
    cat data/lang_nosp_bd/words.txt \
         | awk '{print $1}' | grep -v -E '<eps>|<s>|</s>|#0' | grep -E '^<|^!|^\(|^"|^\)|^,|^-|^\.|^%|^&|^/|^;|^\?' | sort | uniq >> ${nlsyms}_tmp || exit 1;
    cat $bpe_local_dict/words.txt | grep -E '^\{|^\}|^~|^<' >> ${nlsyms}_tmp
    cat ${nlsyms}_tmp | sort | uniq > ${nlsyms}

    # Prepare texts
    #
    (cat data/train_si284/text; cat data/test_dev93/text; cat data/test_eval92/text) | cut -f 2- -d" " > $bpe_local_dict/texts

    tokenise.py $bpe_local_dict/texts $bpe_local_dict/words.txt $bpe_local_dict --nbpe $nbpe --nlsyms $bpe_local_dict/nlsyms

    cat $bpe_local_dict/lexicon.txt | cut -f 2- -d" " | tr " " "\n" | sort | uniq > $bpe_local_dict/nonsilence_phones.txt
    echo "<space>" > $bpe_local_dict/silence_phones.txt
    echo "<space>" > $bpe_local_dict/optional_silence.txt

    utils/prepare_lang.sh --position-dependent-phones false $bpe_local_dict \
                        "<UNK>" ${bpe_local_dict}_tmp data/lang_bpe
    cp -r data/lang_bpe data/lang_bpe_bd
    local/wsj_format_data.sh --lang-suffix "_bpe"
    local/wsj_format_local_lms.sh --lang-suffix "_bpe"
fi

fbankdir=fbank

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: feature extraction and dump"
    for x in train_si284_sp test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        steps/compute_cmvn_stats.sh data/${x}
        utils/fix_data_dir.sh data/${x}
        feat_tr_dir=data/${x}/dump/delta${do_delta}; mkdir -p ${feat_tr_dir}
        dump_utt2spk.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${x} exp/dump_feats/${x} ${feat_tr_dir}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Generating different topologies and token FSTs."
    for topo in $topos; do
        for suffix in $lm_suffixes; do
            prepare_graph.sh --type $topo data/lang_bpe_${suffix} data/local/lang_bpe_${suffix}_${topo}_tmp
        done
    done
fi
