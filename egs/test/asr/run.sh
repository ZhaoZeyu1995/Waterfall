#!/bin/bash

# Copyright 2022 University of Edinburgh (Zeyu Zhao)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./env.sh || exit 1;
. ./cmd.sh || exit 1;

topos="ctc mmictc 2state 2state_blk mmictc_blk"
lm_suffixes="test_tg test_bd_tg test_bd_fg"

nbpe=5000
bpemode=unigram

topo=ctc


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=data/lang_${topo}

local/prepare_dict.sh

utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict \
    "<UNK>" data/local/lang_tmp $lang


#if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    #echo "stage 4: Generating different topologies and token FSTs."
    #for suffix in $lm_suffixes; do
        #for topo in $topos; do
prepare_${topo}.sh $lang
#prepare_graph.sh --type ctc data/lang data/local/lang_bpe_${nbpe}_${suffix}_${topo}_tmp
        #done
    #done
#fi
