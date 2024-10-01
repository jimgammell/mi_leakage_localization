cd ~/Desktop/mi_leakage_localization
python ./src/scripts/adversarial_synthetic_unprotected.py --name adversarial_synthetic_unprotected__big_lr --seed 0 --overwrite &
sleep .1
python ./src/scripts/adversarial_synthetic_shuffling.py --name adversarial_synthetic_shuffling__big_lr --seed 0 --overwrite &
sleep .1
python ./src/scripts/adversarial_synthetic_no_ops.py --name adversarial_synthetic_no_ops__big_lr --seed 0 --overwrite &