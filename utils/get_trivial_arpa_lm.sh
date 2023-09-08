#!/bin/bash

words=$1

# Number of words
N=$(wc -l < $words)
N=$(($N + 3))

# Log probability for each word
log_prob=$(echo "l(1/$N)/l(10)" | bc -l)

# Print ARPA header
echo "\\data\\"
echo "ngram 1=$N"
echo ""
echo "\\1-grams:"

# Print each word with its log probability
while read -r word; do
    printf "%.8f\t%s\n" "$log_prob" "$word"
done < $words

printf "%.8f\t%s\n" "$log_prob" "</s>"
printf "%.8f\t%s\n" "$log_prob" "<s>"
printf "%.8f\t%s\n" "$log_prob" "<unk>"

# Print ARPA footer
echo "\\end\\"

