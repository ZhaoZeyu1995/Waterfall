#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
max_active="7000"
acoustic_scale="6"
beams='16'
latbeam="10"
decode_sets="dev_clean dev_other test_clean test_other"
lang_dir=data/lang_tg
suffix= 


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

graph=${lang_dir}/TLG.fst

# Generate a fake transition model just for mapping transition ids to pdf ids
func/generate_transition_model.sh $lang_dir

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/decode_${decode_set}_latgen_${suffix}
    cp -r $exp_dir/decode_${decode_set} $predict_dir
    for beam in $beams; do 
        for maxac in $max_active; do 
            for acwt in $acoustic_scale; do
                func/decode_fst.sh --nj $nj --max_active $maxac --acoustic_scale $acwt --beam $beam --latbeam $latbeam $data_dir $lang_dir $predict_dir
            done
        done
    done
done

