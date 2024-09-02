# NLCTA: A Natural Language CTA strategy module

## Introduction

Automated build CTA strategy based with several technical indicators combinations.

## Installation

```bash
python -m venv nlcta_env
source nlcta_env/bin/activate

pip install -r requirements.txt
```

## Guide

### We provide several technical indicators

```plaintext
- SMA: Simple Moving Average
- EMA: Exponential Moving Average
- MACD: Moving Average Convergence Divergence
- RSI: Relative Strength Index
- BB: Bollinger Bands
```

You can combine these indicators to create your own strategy, make sure that you follows the rules to correctly call the indicators, and also **check that your combination makes sense**.

- Single SMA with $k=10$ crossing the asset's close price

```plaintext
SMA(10)
```

- Double SMA with $k=10$ and $k=20$ crossing each other

```plaintext
SMA(10, 20)
```

- Single EMA with $k=10$ combined with RSI with $k=14$ for strict buy/sell signals

```plaintext
EMA(10) & RSI(14)
```

### Usage

To customize the combined strategy or backtest settings, you can modify the configuration file `config/combine_test.yaml`.

Inside the `.yaml` file, you can find out there are several sections you can modify:

- `DataAPIYahoo` / `DataAPITAIFEX`: The data API settings, including the asset's ticker, start date, and end date.
- `Strat`: The combined strategy settings, including the indicators and their parameters.
- `Settings`: Initial capital, commission fee, optional stop loss and stop profit thresholds.

### Execute

To execute the whole module, you can run the following command:

```bash
python -m src.main --config_path config/combine_test.yaml --config_type yahoo --plot
```
