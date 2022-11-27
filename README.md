# Waterfall
An ASR toolkit with the freedom of topology

*NOTE: This project is still under development, so it may have some bugs. Currently, it is more like a personal experiment platform for my research.*

## Main Feature

This is a Differentiable WFST (DWFST)-based E2E ASR toolkit (or my personal experiment platform), where CTC is actually a special case in this framework.
This toolkit is mainly based on [k2](https://github.com/k2-fsa/k2) (for DWFST), [PyTorch ](https://pytorch.org/) (for DNN modelling and training), 
and [Kaldi](https://kaldi-asr.org/doc/index.html) (for data preparation and decoding).

Thanks to DWFST, in this toolkit, we may freely define our own topologies of the token FST.

I suppose this toolkit perhaps might be regarded as the basis of the E2E ASR research about topology.

## Installation and Configuration

### Prerequisites

1. kaldi (mainly for data preparation and decoding)
2. miniconda (or anaconda, for environment management)
3. PyTorch and pytorch-lightning (for deep neural network training)
4. k2 (for loss functions based on Differentiable WFST)
5. torchaudio (mainly for wav2vec 2.0), please note that some early versions do not support wav2vec 2.0
6. ESPnet (for SpecAugment and Conformer), you just need to install it with pip.

First of all, you need to install kaldi and create a soft link in tools/.
After that, what you need to do is to create a conda environment with python, 
activate the environment and install pytorch (I have tested the codes with torch 1.11.0), pytorch-lightning, and other dependencies.

The most tricky part is the installation of k2, and you may refer to the [documentation of k2](https://k2-fsa.github.io/k2/) for more details.
In summary, I also recommend you to install k2 with conda.

## Data preparation and Training

There are currently only two DNN models, i.e, wav2vec 2.0 (powered by torchaudio) and conformer (borrowed from ESPnet). 
The recipes are all located at `egs/*/asr*`. Now I mainly work on WSJ and Librispeech, and I am sure that I will add more recipes for other datasets.
In each recipe, you only need to run `prepare.sh` to do data preparation. There are some differences between `asr*` within the same dataset directory, 
and they are mostly introduced in the `readme` there.

After the data preparation, you may run `train.sh` with some configurations to train a wav2vec 2.0 model, 
or `func/train_conformer.sh` plus configurations to train a conformer model.

The details about the configurations of training will be added.

## About Topology

The main feature of this toolkit is that you can define your own topologies freely with almost no limitations.
You may refer to the scripts like `utils/prepare_2state_blk.sh` or `utils/prepare_mmictc_blk.sh` as examples and prepare your own scripts of creating topologies.

## Decoding

The main decoding method here is the Viterbi decoder in kaldi.
You may run the code `decode.sh` or `decode_fst.sh` to decode.
There are actually more decoders available in this toolkit but they are not that mature and still under development.

## Contact

Please feel free to email me (zeyuhongwu1995@gmail.com) should you have any questions and difficulties when using this toolkit 
(perhaps, it is better to call it an experiment platform for now.)




