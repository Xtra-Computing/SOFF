#!/bin/bash

# python -u age.py --ds 'imdbwiki'
mkdir -p result/baseline

python -u age.py -ds 'allagefaces'      2>&1 | tee result/baseline/result_allagefaces
python -u age.py -ds 'appa'             2>&1 | tee result/baseline/result_appa
python -u age.py -ds 'allagefaces_appa' 2>&1 | tee result/baseline/result_allagefaces_appa
python -u age.py -ds 'allagefaces'      -l saves/model_imdb -ee 0 2>&1 | tee result/baseline/result_allagefaces_with_pretrain
python -u age.py -ds 'appa'             -l saves/model_imdb -ee 0 2>&1 | tee result/baseline/result_appa_with_pretrain
python -u age.py -ds 'allagefaces_appa' -l saves/model_imdb -ee 0 2>&1 | tee result/baseline/result_allagefaces_appa_with_pretrain


