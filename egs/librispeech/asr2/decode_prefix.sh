#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
max_active="7000"
acoustic_scale="6"
beams='16'
decode_sets="dev_clean dev_other test_clean test_other"
ac_cost_thresholds="30"
score_based_thresholds="20"
count_based_thresholds="50"
lang_dir=data/lang_tg
suffix= 


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

graph=${lang_dir}/TLG.fst

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/decode_prefix_${decode_set}_${suffix}
    if [ ! -d $predict_dir ]; then
        cp -r $exp_dir/predict_${decode_set} $predict_dir
    fi
    for act in $ac_cost_thresholds; do
        for sbt in $score_based_thresholds; do
            for cbt in $count_based_thresholds; do
                for beam in $beams; do 
                    for maxac in $max_active; do 
                        for acwt in $acoustic_scale; do
                            func/decode_prefix.sh --nj $nj --max_active $maxac --acoustic_scale $acwt --beam $beam --ac_cost_threshold $act --score_based_threshold $sbt --count_based_threshold $cbt $data_dir $lang_dir $predict_dir
                        done
                    done
                done
            done
        done
    done
done

