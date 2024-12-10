#!/bin/bash

cd ~/Desktop/mi_leakage_localization/src
python run_trials.py --dataset synthetic --leakage-type=1o &
python run_trials.py --dataset synthetic --leakage-type=2o &
python run_trials.py --dataset synthetic --leakage-type=12o &
python run_trials.py --dataset synthetic --leakage-type=shuffling &
python run_trials.py --dataset synthetic --leakage-type=no_ops &
python run_trials.py --dataset synthetic --leakage-type=multi_1o &