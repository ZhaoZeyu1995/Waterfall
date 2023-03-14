#!/bin/bash 

. ./path.sh
. ./env.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3

set -e
set -o pipefail
set -u

nj=10
max_active="20"
topo_max_active="10"
acoustic_scale="1.0"
beams='16'
topo_beam='100'
decode_sets="test_yesno"
lang_dir=data/lang_tg
suffix= # can be "" "_soft" or "_hard"


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

graph=${lang_dir}/TLG.fst

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/decode_lg_${decode_set}_${suffix}
    if [ ! -d $predict_dir ]; then
        mkdir -p $predict_dir
        for x in $exp_dir/predict_${decode_set}/*; do
            if [[ $x == *.ark ]]; then
                continue
            fi 
            if [ -f $x ]; then
                cp $x $predict_dir
            fi
        done
    fi
    for beam in $beams; do 
        for maxac in $max_active; do 
            for acwt in $acoustic_scale; do
                func/decode_lg.sh --nj $nj --max_active $maxac --acoustic_scale $acwt --beam $beam --topo_max_active $topo_max_active --topo_beam $topo_beam $data_dir $lang_dir $predict_dir
            done
        done
    done
done

