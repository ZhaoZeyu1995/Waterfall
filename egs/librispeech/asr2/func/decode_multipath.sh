#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode_fst.sh data/test_dev93 data/lang exp/ctc_LV60K_2l/version_3/decode_test_dev93


nj=10
max_active="6000"
acoustic_scale="6"
beam='16'

. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: decode_multipath.sh [options] <data_dir> <lang_dir> <predict_dir>"
  echo "     --nj                        # default: 10, the number of jobs."
  echo "     --max_active                # default: 6000, the max_active value for decode-faster in kaldi."
  echo "     --acoustic_scale            # default: 6, the acoustic_scale value for decode-faster in kaldi."
  echo "     --beam                      # default: 16, the decoding beam for decode-faster in kaldi."
  echo "e.g.:"
  echo " $0 data/test_dev93 data/lang_tg exp/ctc_LV60K_2l/version_3/decode_test_dev93"
  echo " where <predict_dir> should contain a output.1.scp file storing all of the output posterior probabilities."
  echo " unless there is splitN dir storing all split scp files."
  exit 1
fi

data_dir=$1
lang_dir=$2
predict_dir=$3

graph=${lang_dir}/TLG.fst

# split the output.1.scp
if [ ! -d $predict_dir/split$nj ]; then
    if [ -f $predict_dir/output.1.scp ]; then
        run.pl JOB=1:$nj $predict_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $predict_dir/output.1.scp $predict_dir/split$nj/output.JOB.scp
    else
        echo "Cannot find output.1.scp or split$nj in $predict_dir!!! There should be one of them at least!"
    fi
fi


acwt=$acoustic_scale
maxac=$max_active

decode_dir=$predict_dir/acwt_${acwt}-maxac_${maxac}-beam_${beam}
mkdir -p $decode_dir
run.pl JOB=1:$nj $decode_dir/split${nj}/log/decode_viterbi.JOB.log func/decode_multipath.py --max_active $maxac --acoustic_scale $acwt --beam $beam --word_symbol_table ${lang_dir}/words.txt ${graph} $predict_dir/split$nj/output.JOB.scp  $decode_dir/split$nj/hyp.JOB.wrd

for i in $(seq $nj); do
    utils/int2sym.pl -f 2- $lang_dir/words.txt $decode_dir/split$nj/hyp.$i.wrd | cat || exit 1
done > $decode_dir/hyp.wrd.txt || exit 1

post_process_decode.py $decode_dir/hyp.wrd.txt $data_dir/utt2spk > $decode_dir/hyp.wrd.trn

sclite -r $predict_dir/ref.wrd.trn.1 trn -h $decode_dir/hyp.wrd.trn trn -i rm -o all stdout > $decode_dir/results.wrd.txt

