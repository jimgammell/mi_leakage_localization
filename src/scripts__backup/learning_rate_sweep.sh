cd ~/Desktop/mi_leakage_localization
parts=(0 1 2 3 4)
for part in "${parts[@]}"; do
    python ./src/scripts/learning_rate_sweep.py --name "learning_rate_sweep__ii__${part}" --overwrite --seed 0 --part "${part}" &
    sleep .1
done