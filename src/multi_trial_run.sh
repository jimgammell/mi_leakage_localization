#!/bin/bash

cd ~/Desktop/mi_leakage_localization/src
python run_trials.py --dataset DPAv4 --long-lambda-sweep --lambda-idx=0 &
python run_trials.py --dataset DPAv4 --long-lambda-sweep --lambda-idx=1  &
python run_trials.py --dataset DPAv4 --long-lambda-sweep --lambda-idx=2 &
python run_trials.py --dataset DPAv4 --long-lambda-sweep --lambda-idx=3 &
python run_trials.py --dataset DPAv4 --long-lambda-sweep --lambda-idx=4 &