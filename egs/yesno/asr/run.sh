#!/usr/bin/env bash
#

. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./env.sh || exit 1

train_cmd="utils/run.pl"
decode_cmd="utils/run.pl"
nj=4

if [ ! -d waves_yesno ]; then
  wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
  # was:
  # wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
  tar -xvzf waves_yesno.tar.gz || exit 1;
fi

train_yesno=train_yesno
test_base_name=test_yesno

# Data preparation

local/prepare_data.sh waves_yesno
local/prepare_dict.sh
utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict "<UNK>" data/local/lang data/lang
local/prepare_lm.sh

# Feature extraction
fbankdir=fbank
cmvndir=cmvn
for x in train_yesno test_yesno; do 
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
        data/${x} exp/make_fbank/${x} ${fbankdir}
    steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} $cmvndir
    utils/fix_data_dir.sh data/${x}
    dump_utt2spk.sh data/${x} exp/dump/${x} data/${x}/dump/deltafalse
done

for topo in ctc 2state mmictc 2state-1 3state-skip; do
    [ -d data/lang_${topo} ] && rm -rf data/lang_${topo}
    cp -r data/lang data/lang_${topo}
    prepare_${topo}.sh data/lang_${topo}
done



