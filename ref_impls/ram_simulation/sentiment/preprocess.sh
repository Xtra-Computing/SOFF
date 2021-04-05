#!/bin/bash

mkdir -p data
pushd data

mkdir -p imdb
pushd imdb

wget -nc https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

tar -xvzf aclImdb_v1.tar.gz

#!/bin/bash

:> ./imdb_text
:> ./imdb_score

i=0
for f in ./aclImdb/train/pos/* ./aclImdb/test/pos/*; do
    cat "$f" >> ./imdb_text
    echo "" >> ./imdb_text
    echo 'p' >> ./imdb_score
    ((++i))
    echo -ne "\r Parsing positive $i"
done

i=0
for f in ./aclImdb/train/neg/* ./aclImdb/test/neg/*; do
    cat "$f" >> ./imdb_text
    echo "" >> ./imdb_text
    echo 'n' >> ./imdb_score
    ((++i))
    echo -ne "\r Parsing negative $i"
done
popd
#---------------------------------------------

mkdir -p amazon
pushd amazon

tar -xvJf amazon_data.tar.bz2

popd
#---------------------------------------------

popd
python preprocess.py
