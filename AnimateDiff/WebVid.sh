#!/bin/bash
git clone https://github.com/m-bain/webvid.git
cd webvid || exit 2
wget http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_train.csv
pip3 install pandas numpy requests mpi4py
python3 download.py --csv_path results_2M_train.csv --partitions 1 --part 0 --data_dir ./data --processes 8