#!/bin/bash

cd ~/Desktop/mi_leakage_localization/src
python run_trials.py --dataset ASCADv1-variable &
python run_trials.py --dataset DPAv4 &
python run_trials.py --dataset AES_HD &