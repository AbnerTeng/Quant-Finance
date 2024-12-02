#!/bin/bash

strats=("SMA(10)" "SMA(10, 20)" "EMA(10)" "EMA(10, 20)" "MACD(12, 26, 9)" "BB(20, 2, True)" "RSI(14)")
max_idx=$((${#strats[@]} - 1))
random_search_times=100

for i in $(seq 1 $random_search_times); do
    for strat in "${strats[@]}"; do
        random_amount=$(gshuf -i 1-2 -n 1)
        random_numbers=$(gshuf -i 0-$max_idx -n $random_amount)
        selected_strats=()

        for num in $random_numbers; do
            selected_strat="${strats[$num]}"
            selected_strats+=("\"$selected_strat\"")
        done

        echo "Randomly selected $random_amount strategies"

        yaml_array=$(printf ", %s" "${selected_strats[@]}")
        yaml_array="[${yaml_array:2}]"

        echo "Updating YAML with strategies: $yaml_array"

        yq eval ".Strat = $yaml_array" -i config/combine_test.yaml
        python -m src.rolling_main --config_path config/combine_test.yaml --config_type yahoo --trials 3
    done
done
