#!/usr/bin/env python3

from tokenizers.trainers import BpeTrainer
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
import argparse









def get_parser():
    parser = argparse.ArgumentParser(
        description='This scrip utilises the tokenisers to extract BPE')
    parser.add_argument(
        'texts', help='The training text for BPE extractor, which can be a comma-separated string, for example data/train_text,data/dev_text,data/eval_text', type=str)
    parser.add_argument('words', help='A file contains the list of all words that need encoding by the BPE model', type=str)
    parser.add_argument('outdir', help='The output directory for the BPE model and the encoded results for the word list (i.e., the lexicon.txt)', type=str)

    parser.add_argument('--nbpe', help='The number of BPE tokens', default=5000, type=int)
    parser.add_argument(
        '--nlsyms', help='non-language symbols list', default=None, type=str)


    return parser

def main(args):
    nlsyms = []
    if args.nlsyms:
        with open(args.nlsyms) as f:
            for line in f:
                symbol = line.strip().split()[0]
                nlsyms.append(symbol)
    pass

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    special_tokens = nlsyms if args.nlsyms else ['<UNK>']
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=args.nbpe)
    tokenizer.pre_tokenizer = Whitespace()

    files = args.texts.split(',')
    tokenizer.train(files, trainer)


    tokenizer.save(os.path.join(args.outdir, 'bpe-model.json'))

    with open(args.words) as f, open(os.path.join(args.outdir, 'lexicon.txt'), 'w') as g:
        gc = ''
        for line in f:
            word = line.strip().split()[0]
            output = tokenizer.encode(word)
            gc += ('%s %s\n' % (word, ' '.join(output.tokens)))
        g.write(gc)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)
