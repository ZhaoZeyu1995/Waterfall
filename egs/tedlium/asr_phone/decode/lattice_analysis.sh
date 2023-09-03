#!/bin/bash 

. ./env.sh || exit 1
. ./path.sh || exit 1
. ./cmd.sh || exit 1

# Decoding with FST-based decoder
# e.g. decode_fst.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
max_active="2000"
acoustic_scale="1.0"
search_beam="32.0"
latbeams="8.0"
decode_sets="test"
lang_dir=data/lang_tg
suffix= 
predict_suffix= 



. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/lattice_${decode_set}_${suffix}
    for latbeam in $latbeams; do 
        for maxac in $max_active; do 
            for acwt in $acoustic_scale; do
                func/lattice_analysis.sh --nj ${nj} --search-beam ${search_beam} --acoustic-scale ${acwt} --lattice-beam ${latbeam} --max-active ${maxac} $data_dir ${lang_dir} $exp_dir/predict_${decode_set}${predict_suffix} ${predict_dir}
            done
        done
    done
done
