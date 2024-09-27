cd ~/Desktop/mi_leakage_localization
python ./src/scripts/supervised_synthetic_unprotected.py --name synthetic_unprotected_supervised_htune --seed 0 --overwrite &
sleep .1
python ./src/scripts/supervised_synthetic_boolean_masking.py --name synthetic_boolean_masking_supervised_htune --seed 0 --overwrite &
sleep .1
python ./src/scripts/supervised_synthetic_no_ops.py --name synthetic_no_ops_supervised_htune --seed 0 --overwrite &
sleep .1
python ./src/scripts/supervised_synthetic_shuffling.py --name synthetic_shuffling_supervised_htune --seed 0 --overwrite &