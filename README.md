# Quant-Finance

Qunatitative Trading Space

> **Note**: This repo is also for all my mentees in the department of Algorithm Trading, TMBA.

Author:

- Yu-Chen (Abner) Den
- Tzu-Hao (Howard) Liu

## Folders

### CTA (Ongoing)

Single asset CTA trading strategy and backtesting system, including various data collection APIs.

*Full documentation can be found in the `CTA/docs` folder*

**Usage:**

- Create environment
  - venv

    ```plaintext
    python3 -m venv your_venv
    source your_venv/bin/activate

    pip3 install -r requirements.txt
    ```

  - Docker (On-going)

    ```plaintext
    docker build -t nlcta .
    docker run -it nlcta
    ```

- Change the `config/combine_test.yaml` file to your own settings (including API keys).
- Run the `main.py` file.

```plaintext
python3 -m src.main
```

### BetterRSP / ETF\_entry (Ongoing)

A better RSP strategy for ETFs that utilize machine learning models to predict the confidence score of next-month entry point

### HWs

Homeworks for my mentees @ TMBA.

### Learning to Rank on Portfolio Construction (LR\_rank)

Instead of classifying portfolio return into 10 independent classes, we rank those returns because we want the relationships between them. 

### Options

Untidy folder full of options payoff diagrams and strategies. (I don't want to clean it up.)

## Start

First create a virtual environment and activate it.

```plaintext
python3 -m venv your_venv
source your_venv/bin/activate
```

Then install the required packages.

```plaintext
pip3 install -r requirements.txt
```
