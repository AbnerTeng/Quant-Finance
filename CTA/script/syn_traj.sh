sp_sl_pair=("-0.99 0.99" "-0.01 0.05" "-0.02 0.10" "-0.05 0.10" "-0.05 0.05" "-0.10 0.15")

# strat="src.indicators.BB"
# params=(20)

# for pair in "${sp_sl_pair[@]}"; do
#     echo "Updating config with strat: $strat and pair: $pair"
#     sl_thres=$(echo $pair | awk '{print $1}')
#     sp_thres=$(echo $pair | awk '{print $2}')

#     param_list=$(printf ", %s" "${params[@]}")
#     param_list="[${param_list:1}]"

#     yq eval ".Settings.sl_thres = $sl_thres | .Settings.sp_thres = $sp_thres" -i config/gen_strat.yaml
#     yq eval ".Class.strat = \"$strat\" | .Class.params = $param_list" -i config/gen_strat.yaml

#     python -m src.rolling_main --data_source self
# done


# strat="src.indicators.rsi.RSI"
# params=(14)

# for pair in "${sp_sl_pair[@]}"; do
#     echo "Updating config with strat: $strat and pair: $pair"
#     sl_thres=$(echo $pair | awk '{print $1}')
#     sp_thres=$(echo $pair | awk '{print $2}')

    # param_list=$(printf ", %s" "${params[@]}")
    # param_list="[${param_list:1}]"

#     yq eval ".Settings.sl_thres = $sl_thres | .Settings.sp_thres = $sp_thres" -i config/gen_strat.yaml
#     yq eval ".Class.strat = \"$strat\" | .Class.params = $param_list" -i config/gen_strat.yaml

#     python -m src.rolling_main --data_source self
# done


strat="src.indicators.ma.SMA"
params=(8 21)

for pair in "${sp_sl_pair[@]}"; do
    echo "Updating config with strat: $strat and pair: $pair"
    sl_thres=$(echo $pair | awk '{print $1}')
    sp_thres=$(echo $pair | awk '{print $2}')

    param_list=$(printf ", %s" "${params[@]}")
    param_list="[${param_list:1}]"

    yq eval ".Settings.sl_thres = $sl_thres | .Settings.sp_thres = $sp_thres" -i config/gen_strat.yaml
    yq eval ".Class.strat = \"$strat\" | .Class.params = $param_list" -i config/gen_strat.yaml

    python -m src.rolling_main --data_source self
done


# strat="src.indicators.ma.MACD"
# params=(12 26 9)

# for pair in "${sp_sl_pair[@]}"; do
#     echo "Updating config with strat: $strat and pair: $pair"
#     sl_thres=$(echo $pair | awk '{print $1}')
#     sp_thres=$(echo $pair | awk '{print $2}')

#     param_list=$(printf ", %s" "${params[@]}")
#     param_list="[${param_list:1}]"

#     yq eval ".Class.strat = \"$strat\" | .Class.params = $param_list" -i config/gen_strat.yaml
#     yq eval ".Settings.sl_thres = $sl_thres | .Settings.sp_thres = $sp_thres" -i config/gen_strat.yaml

#     python -m src.rolling_main --data_source self
# done
