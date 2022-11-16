#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
max_active="2000"
acoustic_scale="0.8"
beams='16'
decode_sets="test_dev93 test_eval92"
lang_dir=data/lang_tg
suffix= 


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

graph=${lang_dir}/TLG.fst

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/decode_tlg_mp_${decode_set}_${suffix}
    if [ ! -d $predict_dir ]; then
        cp -r $exp_dir/predict_${decode_set} $predict_dir
    fi
    for beam in $beams; do 
        for maxac in $max_active; do 
            for acwt in $acoustic_scale; do
                func/decode_tlg_mp.sh --nj $nj --max_active $maxac --acoustic_scale $acwt --beam $beam $data_dir $lang_dir $predict_dir
            done
        done
    done
done

