#!/bin/bash

mkdir -p data

curl -L 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar' -o data/'imdb_crop.tar'

curl -L 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar' -o data/'wiki_crop.tar'

curl -L 'https://www.dropbox.com/s/a0lj1ddd54ns8qy/All-Age-Faces Dataset.zip' -o data/'All-Age-Faces Dataset.zip'

curl -L 'http://158.109.8.102/AppaRealAge/appa-real-release.zip' -o data/'appa-real-release.zip'

pushd data

tar -xvf imdb_crop.tar
tar -xvf wiki_crop.tar
unzip 'All-Age-Faces Dataset.zip'
unzip 'appa-real-release.zip'

popd

python preprocess.py -r -f

