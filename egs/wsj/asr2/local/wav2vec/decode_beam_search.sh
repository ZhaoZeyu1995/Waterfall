#!/bin/bash

. ./path.sh
. ./cmd.sh

set_names="test_dev93 test_eval92"
exp_dir="exp/ctc_LV60K_2l/version_3"
nj=10
prune="-20"
beam_prunes="-10"
beam_sizes="20"
betas="0"
alphas="0.000 0.500"
arpa_lm=/disk/scratch3/zzhao/data/wsj/wsj.arpa

. utils/parse_options.sh


for set_name in ${set_names}; do
    output_dir=$exp_dir/decode_$set_name
    run.pl JOB=1:$nj $output_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $output_dir/output.1.scp $output_dir/split$nj/output.JOB.scp
done

for set_name in ${set_names}; do
    data_dir=data/$set_name
    lang_dir=data/lang
    decode_dir=${exp_dir}/decode_${set_name}
    for alpha in ${alphas}; do
        for beta in ${betas}; do
            for beam_size in ${beam_sizes}; do
                for beam_prune in ${beam_prunes}; do
                    result_dir=${decode_dir}/alpha_${alpha}_beta_${beta}_beamsize_${beam_size}_beamprune_${beam_prune}
                    run.pl JOB=1:${nj} ${result_dir}/log/decode.JOB.log \
                        time decode.py \
                        --predict_path ${decode_dir}/split${nj} \
                        --data_dir ${data_dir} \
                        --lang_dir ${lang_dir} \
                        --jid JOB \
                        --prune ${prune} \
                        --beam_prune ${beam_prune} \
                        --beam_size ${beam_size} \
                        --arpa_lm ${arpa_lm} \
                        --beta ${beta} \
                        --alpha ${alpha}

                    cp ${decode_dir}/ref.wrd.trn.1 ${result_dir}/ref.wrd.trn
                    for i in $(seq 1 $nj); do
                        cat ${result_dir}/hyp.wrd.trn.${i}
                    done > ${result_dir}/hyp.wrd.trn

                    sclite -r ${result_dir}/ref.wrd.trn trn -h ${result_dir}/hyp.wrd.trn trn -i rm -o all stdout > ${result_dir}/results.wrd.txt
                done
            done
        done
    done
done
