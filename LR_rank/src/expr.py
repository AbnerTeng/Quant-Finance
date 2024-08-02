# %%
import pandas as pd

# %%
df = pd.read_csv('../data/usmo.csv', encoding='utf-8')
unq_name = df['stock_id'].unique()
# train, test = df[df['Date'] < '2021-01-01'], df[df['Date'] >= '2021-01-01']
# %%
def calc_month_perf(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly return
    can add other indicators later
    """
    data['ret'] = [0] + list(data['close'][1:].values / data['close'][:-1].values - 1)
    data['next_ret'] = list(data['ret'][1:].values) + [0]
    return data

# %%
df = df.groupby('stock_id').apply(calc_month_perf)
# %%
train = train.groupby('stock_id').apply(calc_month_perf)
test = test.groupby('stock_id').apply(calc_month_perf)
# %%
df.dropna(inplace=True)
df = df.droplevel(1)
# %%
from typing import Dict
import numpy as np


def rank_func(ret: pd.Series, method: str) -> pd.Series:
    """
    Rank function for next_ret
    Not figuring out which one is better
    Or I can just try all methods
    """
    if method == "minmax":
        return (ret - ret.min()) / (ret.max() - ret.min())
    elif method == "order":
        return ret.rank(method='min')
    elif method == "quantile":
        return pd.qcut(ret.rank(method='first'), 10, labels=False) / 10 + 0.1
    elif method == "softmax":
        return np.exp(ret) / np.exp(ret).sum()
    else:
        raise ValueError("Method not supported")


def rank_across_stocks(data: pd.DataFrame, method: str) -> Dict[str, pd.Series]:
    rank = {}
    dates = data.Date.unique()
    for date in dates:
        subdata = data[data.Date == date]
        rank[date] = rank_func(subdata['next_ret'], method)
    return rank

# %%
rank_data = rank_across_stocks(df, "quantile")
# %%
train = train.droplevel(1) ## need to delete this step in the future
tr_rank = rank_across_stocks(train, "quantile")
# %%
def merge_to_data(map: Dict[str, pd.Series], data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the rank to data
    """
    for date, rank in map.items():
        data.loc[data.Date == date, 'label'] = rank
    return data
# %%
df = merge_to_data(rank_data, df)
# %%
train = merge_to_data(tr_rank, train)
# %%
feature = df.drop(columns=['adj close', 'ret', 'next_ret'])
# %%
from tqdm import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error


model = XGBRegressor(random_state=42)
feature = feature.sort_values(by="Date")
unq_date = feature['Date'].unique()
WINDOW_SIZE = 60
mae = []
new_testdf = pd.DataFrame()
for i in tqdm(range(len(unq_date) - WINDOW_SIZE + 1)):
    if i + WINDOW_SIZE >= len(unq_date):
        break
    train = feature[feature['Date'].isin(unq_date[i:i+WINDOW_SIZE])]
    ntrain = train.drop(columns=['Date', 'stock_id'])
    train_x, train_y = ntrain.drop(columns=['label']), train['label']
    model.fit(train_x, train_y)
    test = feature[feature['Date'] == unq_date[i+WINDOW_SIZE]]
    ntest = test.drop(columns=['Date', 'stock_id'])
    test_x, test_y = ntest.drop(columns=['label']), test['label']
    prediction = model.predict(test_x)
    test['pred'] = prediction
    mae.append(mean_absolute_percentage_error(test_y, prediction))
    new_testdf = pd.concat([new_testdf, test])

print(np.mean(mae))

# %%
def run(n, m, new_testdf, df):
    grp = new_testdf.groupby('Date')
    for day in tqdm(new_testdf['Date'].unique()):
        subdf = grp.get_group(day)
        sorted_indices = subdf['pred'].sort_values(ascending=False).index
        highest_n = sorted_indices[:n]
        if m > 0:
            lowest_n = sorted_indices[-m:]
        else:
            lowest_n = []

        new_testdf.loc[new_testdf['Date'] == day, 'l_s'] = 0
        new_testdf.loc[
            (new_testdf['Date'] == day) & (new_testdf.index.isin(highest_n)), 'l_s'
        ] = 1
        new_testdf.loc[
            (new_testdf['Date'] == day) & (new_testdf.index.isin(lowest_n)), 'l_s'
        ] = -1
        new_testdf.loc[
            (new_testdf['Date'] == day) & (new_testdf['stock_id'].isin(sorted_indices.to_list())), 'next_ret'
        ] = df[(df['Date'] == day) & (df['stock_id'].isin(sorted_indices.to_list()))]['next_ret']

    new_testdf['total_ret'] = new_testdf['next_ret'] * new_testdf['l_s']

    ret = []
    for day in new_testdf['Date'].unique():
        ret.append(new_testdf[new_testdf['Date'] == day]['total_ret'].sum() / (n + m))
    return ret
# %%
rets, crets = [], []
for n in range(2, 7):
    ret = run(n, 0, new_testdf, df)
    cret = np.array(ret).cumsum()
    rets.append(ret)
    crets.append(cret)
# %%
def metrics(r):
    cumret = np.array(r).cumsum()[-1]
    ann_sharpe = np.mean(r) / np.std(r) * np.sqrt(12)
    peak = np.maximum.accumulate(np.array(r))
    drawdown = -(np.array(r) - peak) / (1 + peak)
    mdd = np.max(drawdown)
    calmar3y = np.mean(r) * 36 / np.max(drawdown)
    winrate = np.where(np.array(r) > 0)[0].shape[0] / len(r)
    return (
        f"Cumret: {cumret}, Sharpe: {ann_sharpe}, Mdd: {mdd}, Calmar: {calmar3y}, WR: {winrate}"
    )

import matplotlib.pyplot as plt
bench_ret = []
for day in new_testdf['Date'].unique():
    bench_ret.append(df[df['Date'] == day]['next_ret'].mean())

plt.plot(np.array(bench_ret).cumsum(), color="blue", label="BnH")
labels = [f"LR_rank: ({n}, 0)" for n in range(2, 7)]
plt.plot(np.array(crets).T, label=labels)
plt.legend()
plt.show()

print(f"BnH metrics: {metrics(bench_ret)}")
print(f"LR_rank metrics: {metrics(ret)}")
# %%

# %%
from xgboost import XGBRegressor

def trainstep(feat: pd.DataFrame, label: pd.Series) -> XGBRegressor:
    """
    training steps
    """
    model = XGBRegressor()
    model.fit(feat, label)
    return model

def teststep(model: XGBRegressor, feat: pd.DataFrame) -> pd.Series:
    """
    testing steps
    """
    y_pred = model.predict(feat)
    return y_pred
# %%
feature = train.drop(columns=['Date', 'adj close', 'stock_id', 'ret', 'next_ret', 'label'])
test_feature = test.drop(columns=['Date', 'adj close', 'stock_id', 'ret', 'next_ret'])
label = train['label']
trained_mdl = trainstep(feature, label)
predict_val = teststep(trained_mdl, test_feature)
# %%
def mean_squared_error(true, pred):
    return 1 / len(true) * np.sum((true - pred) ** 2)
# %%
test = test.droplevel(1)
ts_rank = rank_across_stocks(test, "quantile")
test = merge_to_data(ts_rank, test)
print(mean_squared_error(test['label'], predict_val))
# %%
# from CTA.src.get_data import DataAPI
# from CTA.src.utils.data_utils import (
#     get_self,
#     transfer_colnames
# )

# cfg = get_self('CTA/config/fetcher.yaml')
# fetcher = DataAPI(**cfg['DataAPIYahoo'])
# data = fetcher.fetch()
# data = transfer_colnames(data)
# data.to_csv("data/usmo.csv")

# %%
from sklearn.datasets import make_classification
import numpy as np

import xgboost as xgb

# Make a synthetic ranking dataset for demonstration
seed = 1994
X, y = make_classification(random_state=seed)
rng = np.random.default_rng(seed)
n_query_groups = 3
qid = rng.integers(0, n_query_groups, size=X.shape[0])

# Sort the inputs based on query index
sorted_idx = np.argsort(qid)
X = X[sorted_idx, :]
y = y[sorted_idx]
qid = qid[sorted_idx]
# %%
ranker = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg", lambdarank_pair_method="topk")
ranker.fit(X, y, qid=qid)
scores = ranker.predict(X)
sorted_idx = np.argsort(scores)[::-1]
# Sort the relevance scores from most relevant to least relevant
scores = scores[sorted_idx]
# %%
print(scores)
# %%
np.where(qid[sorted_idx] == 2)[0].shape
# %%
import pandas as pd

df = pd.read_csv('../data/prediction.csv', encoding='utf-8', index_col=0)

# %%
unique_dates = df['Date'].unique()

long_rets = []
short_rets = []

for idx, date in enumerate(unique_dates):
    if idx == len(unique_dates) - 1:
        break
    long_stocks = df[df['Date'] == date].sort_values('pred', ascending=False).index[:2].to_list()
    short_stocks = df[df['Date'] == date].sort_values('pred', ascending=True).index[:2].to_list()
    subdf = df[df['Date'] == unique_dates[idx + 1]]
    subtest = test[test['Date'] == unique_dates[idx + 1]]
    long_close = subdf.loc[long_stocks, 'close']
    short_close = subdf.loc[short_stocks, 'close']
    long_ret = subtest.loc[long_stocks, 'next_ret']
    short_ret = subtest.loc[short_stocks, 'next_ret']
    long_rets.append(long_ret.values)
    short_rets.append(short_ret.values)

print(np.mean(np.array(long_rets), axis=1))
print(np.mean(np.array(short_rets), axis=1))
# %%
merge_ret = np.mean(np.array(long_rets), axis=1) - np.mean(np.array(short_rets), axis=1)
print(merge_ret)
# %%
import matplotlib.pyplot as plt
plt.plot(merge_ret.cumsum(), "o")
# %%