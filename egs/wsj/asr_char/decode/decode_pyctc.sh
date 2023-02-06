#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with prefix search decoder
# e.g. decode_pyctc.sh --suffix "_soft" exp/ctc_LV60K_2l/version_3


nj=10
alphas="0.5"
betas="1.5"
beam_widths='50'
beam_prune_logps="-10"
token_min_logps="-5"
lm=data/local/nist_lm/lm_tg.arpa

decode_sets="test_dev93 test_eval92"
lang_dir=data/lang_tg
suffix= 


. ./utils/parse_options.sh

exp_dir=$1 # e.g. exp/ctc_LV60K_2l/version_3


for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    predict_dir=$exp_dir/decode_pyctc_${decode_set}_${suffix}
    if [ ! -d $predict_dir ]; then
        cp -r $exp_dir/predict_${decode_set} $predict_dir
    fi
    for beam_width in $beam_widths; do 
        for alpha in $alphas; do 
            for beta in $betas; do
                for beam_prune_logp in $beam_prune_logps; do
                    for token_min_logp in $token_min_logps; do 
                        func/decode_pyctc.sh --nj $nj --alpha $alpha --beta $beta --beam_width $beam_width --beam_prune_logp $beam_prune_logp --token_min_logp $token_min_logp --lm $lm $data_dir $lang_dir $predict_dir
                    done
                done
            done
        done
    done
done

