#!/bin/bash
#


lang=$1

fstdraw --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L.fst | dot -Tpdf > $lang/L.pdf
fstdraw --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L_disambig.fst | dot -Tpdf > $lang/L_disambig.pdf
fstdraw --isymbols=$lang/k2/tokens.txt --osymbols=$lang/k2/phones.txt $lang/k2/T.fst | dot -Tpdf > $lang/k2/T.pdf
fstdraw --isymbols=$lang/tokens_disambig.txt --osymbols=$lang/phones.txt $lang/T.fst | dot -Tpdf > $lang/T.pdf
