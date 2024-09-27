cd ~/Desktop/mi_leakage_localization
python ./src/scripts/supervised_aes_hd.py --name aes_hd_supervised_htune --seed 0 --overwrite &
python ./src/scripts/supervised_dpav4.py --name dpav4_hd_supervised_htune --seed 0 --overwrite &
python ./src/scripts/supervised_aes_rd.py --name aes_rd_supervised_htune --seed 0 --overwrite &
python ./src/scripts/supervised_ascadv1f.py --name ascadv1f_supervised_htune --seed 0 --overwrite &