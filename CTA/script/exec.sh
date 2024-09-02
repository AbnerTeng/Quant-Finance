#!/bin/bash

strats=("SMA(10)" "SMA(10, 20)" "MACD(12, 26, 9)" "BB(20, True)" "RSI(14)")
max_idx=$((${#strats[@]} - 1))
random_search_times=10

for i in $(seq 1 $random_search_times); do
    amount=1
    for strat in "${strats[@]}"; do
        random_numbers=$(gshuf -i 0-$max_idx -n $amount)
        selected_strats=()

        for num in $random_numbers; do
            selected_strat="${strats[$num]}"
            selected_strats+=("\"$selected_strat\"")
        done

        yaml_array=$(printf ", %s" "${selected_strats[@]}")
        yaml_array="[${yaml_array:2}]"

        echo "Updating YAML with strategies: $yaml_array"

        yq eval ".Strat = $yaml_array" -i config/combine_test.yaml
        python -m src.main --config_path config/combine_test.yaml --config_type yahoo
        amount=$((amount+1))
    done
done

