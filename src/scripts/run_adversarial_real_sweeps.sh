cd ~/Desktop/mi_leakage_localization
python ./src/scripts/adversarial_ascadv1f.py --name adversarial_ascadv1f --seed 0 --overwrite &
sleep .1
python ./src/scripts/adversarial_dpav4.py --name adversarial_dpav4 --seed 0 --overwrite &
sleep .1
python ./src/scripts/adversarial_aes_hd.py --name adversarial_aes_hd --seed 0 --overwrite &
sleep .1
python ./src/scripts/adversarial_aes_rd.py --name adversarial_aes_rd --seed 0 --overwrite &