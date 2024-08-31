# Naming Rules

## Simple Moving Average (SMA)

```plaintext
CLASS SMA(k1, k2=None)
```

Applies a simple moving average strategy, where $k_1$ is the period of the moving average, and the strategy will return a buy signal when the asset's close price crosses the moving average from below, and a sell signal when the asset's close price crosses the moving average from above.

$$\text{SMA}_k = \dfrac{1}{k}\sum_{i=n-k+1}^{n} \text{Close}_{i}$$

Furthermore, if $k_2$ is provided, the strategy will return a buy signal when the short moving average crosses the long moving average from below, and a sell signal when the short moving average crosses the long moving average from above.

## Expotential Moving Average (EMA)

```plaintext
CLASS EMA(k1, k2=None)
```

Same as SMA, but with an exponential moving average.

$$\text{EMA}_k = \dfrac{2}{k+1} \times \text{Close}_{n} + \left(1 - \dfrac{2}{k+1}\right) \times \text{EMA}_{n-1}$$

## Moving Average Convergence Divergence (MACD)

```plaintext
CLASS MACD(k1, k2, k3)
```

To build a MACD strategy, you **must** provide three parameters: $k_1$, $k_2$, and $k_3$. The strategy will return a buy signal when the MACD line crosses the signal line from below, and a sell signal when the MACD line crosses the signal line from above.

$$\text{MACD} = \text{EMA}_{k1} - \text{EMA}_{k2}$$

$$\text{Signal} = \text{EMA}_{k3}(\text{MACD})$$

## Relative Strength Index (RSI)

```plaintext
CLASS RSI(k1)
```

To build an RSI strategy, you **must** provide one parameter: $k_1$ which is the period of the RSI. The strategy will return a buy signal when the RSI index is above 70 and a sell signal when the RSI index is below 30.

$$
\text{RSI} = 100 - \dfrac{100}{1 + \text{RS}} \\
\text{RS} = \dfrac{\text{Average Gain}}{\text{Average Loss}}
$$

## Bollinger Bands (BBANDS)

```plaintext
CLASS BBANDS(k1, reverse=False)
```

To build a Bollinger Bands strategy, you **must** provide one parameter: $k_1$, which is the length of the moving average to construct upper bound and lower bound. The strategy will return in two types

1. `reverse = False`: Momentum strategy, buy signal where the asset's close price crosses the upper bound from below, and a sell signal when the asset's close price crosses the moving average from above.

2. `reverse = True`: Reversion strategy, buy signal where the asset's close price crosses the lower bound from below, and a sell signal when the asset's close price crosses the moving average from below.

$$
\text{Upper} = \text{SMA}_k + 2 \times \text{STD}_k \\
\text{Lower} = \text{SMA}_k - 2 \times \text{STD}_k
$$