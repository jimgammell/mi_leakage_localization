cd ~/Desktop/mi_leakage_localization
parts=(0 1 2 3 4 5 6 7 8 9)
for part in "${parts[@]}"; do
    python ./src/scripts/learning_rate_sweep.py --name "learning_rate_sweep__${part}" --overwrite --seed 0 --part "${part}" &
done