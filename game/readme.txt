1. Please run the Python scripts to prepare the data with the following orders:
(1) clean_ign_game.py
(2) clean_steam_game.py
(3) sample_steam_interact.py
(4) align.py
(5) negative_sample_align_interact.py

Note that we have uploaded the cleaned version of steam and ign data, so actually there is no need to run the scripts of (1) and (2).

2. Run the experiments
python train_align.py --setting=one
python train_align.py --setting=two