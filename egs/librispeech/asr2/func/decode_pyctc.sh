#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh data/test_dev93 data/lang exp/ctc_LV60K_2l/version_3/decode_test_dev93


nj=10
alpha="0.5"
beta="1.5"
beam_width='50'
beam_prune_logp="-10"
token_min_logp="-5"
lm=

. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: decode_pyctc.sh [options] <data_dir> <lang_dir> <predict_dir>"
  echo "     --nj                        # default: 10, the number of jobs."
  echo "e.g.:"
  echo " $0 data/test_dev93 data/lang_tg exp/ctc_LV60K_2l/version_3/decode_test_dev93"
  echo " where <predict_dir> should contain a output.1.scp file storing all of the output posterior probabilities."
  echo " unless there is splitN dir storing all split scp files."
  exit 1
fi

data_dir=$1
lang_dir=$2
predict_dir=$3


# split the output.1.scp
if [ ! -d $predict_dir/split$nj ]; then
    if [ -f $predict_dir/output.1.scp ]; then
        run.pl JOB=1:$nj $predict_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $predict_dir/output.1.scp $predict_dir/split$nj/output.JOB.scp
    else
        echo "Cannot find output.1.scp or split$nj in $predict_dir!!! There should be one of them at least!"
    fi
fi

decode_dir=$predict_dir/alpha_${alpha}-beta_${beta}-beam_width_${beam_width}-beam_prune_${beam_prune_logp}-token_min_logp_${token_min_logp}
mkdir -p $decode_dir
if [ -z $lm ]; then
    run.pl JOB=1:$nj $decode_dir/split${nj}/log/decode_pyctc.JOB.log func/decode_pyctc.py --alpha $alpha --beta $beta --beam_width $beam_width --beam_prune_logp ${beam_prune_logp} --token_min_logp ${token_min_logp} --word_symbol_table ${lang_dir}/words.txt $predict_dir/split$nj/output.JOB.scp  $decode_dir/split$nj/hyp.JOB.wrd $lang_dir/k2/tokens.txt 
else
    run.pl JOB=1:$nj $decode_dir/split${nj}/log/decode_pyctc.JOB.log func/decode_pyctc.py --alpha $alpha --beta $beta --beam_width $beam_width --beam_prune_logp ${beam_prune_logp} --token_min_logp ${token_min_logp} --lm $lm --word_symbol_table ${lang_dir}/words.txt $predict_dir/split$nj/output.JOB.scp  $decode_dir/split$nj/hyp.JOB.wrd $lang_dir/k2/tokens.txt 
fi

for i in $(seq $nj); do
    cat $decode_dir/split$nj/hyp.$i.wrd || exit 1;
done > $decode_dir/hyp.wrd.txt || exit 1

post_process_decode.py $decode_dir/hyp.wrd.txt $data_dir/utt2spk > $decode_dir/hyp.wrd.trn

sclite -r $predict_dir/ref.wrd.trn.1 trn -h $decode_dir/hyp.wrd.trn trn -i rm -o all stdout > $decode_dir/results.wrd.txt

