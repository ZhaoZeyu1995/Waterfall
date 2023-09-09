#!/bin/bash 

. ./path.sh
. ./env.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
max_active="2000"
acoustic_scale="1.0"
beams='16'
decode_sets="dev test"
lang_dir=data/lang_tg
suffix= 
predict_suffix= 


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

graph=${lang_dir}/TLG.fst

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    decode_dir=$exp_dir/decode_${decode_set}
    predict_dir=$exp_dir/predict_${decode_set}
    if [ ! -z $suffix ]; then
        decode_dir=${decode_dir}_${suffix}
    fi
    if [ ! -z $predict_suffix ]; then
        decode_dir=${decode_dir}_${predict_suffix}
        predict_dir=${predict_dir}_${predict_suffix}
    fi
    if [ ! -d $decode_dir ]; then
        mkdir -p $decode_dir
    fi
    for f in ${predict_dir}/*; do
        if [[ $f == *.ark ]]; then
            continue
        fi
        [ -f $f ] || continue
        [ -f $f ] && [ ! -f $decode_dir/$(basename $f) ] && cp $f $decode_dir
    done
    for beam in $beams; do 
        for maxac in $max_active; do 
            for acwt in $acoustic_scale; do
                func/decode_faster.sh --nj $nj --max_active $maxac --acoustic_scale $acwt --beam $beam $data_dir $lang_dir $decode_dir
            done
        done
    done
done

