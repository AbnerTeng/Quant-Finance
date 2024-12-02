strats=("SMA(10)" "SMA(10, 20)" "EMA(10)" "EMA(10, 20)" "MACD(12, 26, 9)" "BB(20, 2, True)" "RSI(14)")
max_idx=$((${#strats[@]} - 1))


random_amount=$((RANDOM % 2 + 1))
random_numbers=()

for ((i=0; i<random_amount; i++)); do
    random_numbers+=($((RANDOM % (max_idx + 1))))
done

selected_strats=()


for num in ${random_numbers[@]}; do
    selected_strat="${strats[$num]}"
    selected_strats+=("\"$selected_strat\"")
done

echo "Randomly selected $random_amount strategies"

yaml_array=$(printf ", %s" "${selected_strats[@]}")
yaml_array="[${yaml_array:2}]"

echo "Updating YAML with strategies: $yaml_array"
