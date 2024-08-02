"""
Learning to Rank for Portfolio Construction
"""
import warnings
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score, mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold
from xgboost import (
    XGBRanker,
    XGBRegressor
)
from tqdm import tqdm
from .utils.data_utils import (
    load_data,
    merge_to_data
)
from .utils.metric_utils import calc_month_perf
from .ranker import Ranker
from .trainer import Trainer
from .portfolio import Portfolio

warnings.filterwarnings("ignore")

def parse_args() -> ArgumentParser:
    """
    parsing arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", "-cp", type=str, default="config/test.yaml"
    )
    parser.add_argument(
        "--model_type", "-mdt", type=str, default="xgbrg"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_data("data/usmo.csv")
    cfg = load_data(args.config_path)
    unq_name = df['stock_id'].unique()
    df = df.groupby('stock_id').apply(calc_month_perf)
    df.dropna(inplace=True)
    df = df.droplevel(1)
    ranker = Ranker(cfg["rank_method"])
    rank_data = ranker.rank_across_stocks(df)
    df = merge_to_data(rank_data, df)
    org_test = df[df['Date'] >= cfg["thres"]]
    feature = df.drop(columns=['adj close', 'ret', 'next_ret'])
    train, test = feature[feature['Date'] < cfg["thres"]], feature[feature['Date'] >= cfg["thres"]]

    if args.model_type.endswith("rg"):
        model = XGBRegressor(random_state=42)
        feature = feature.sort_values(by="Date")
        unq_date = feature['Date'].unique()
        WINDOW_SIZE = 60
        mae = []
        pred_dict = {}
        pred_dict["stock_id"] = feature[feature['Date'] == unq_date[0]].index.to_list()
        for i in tqdm(range(len(unq_date) - WINDOW_SIZE + 1)):
            if i + WINDOW_SIZE >= len(unq_date):
                break
            train= feature[feature['Date'].isin(unq_date[i:i+WINDOW_SIZE])]
            train.drop(columns=['Date', 'stock_id'], inplace=True)
            train_x, train_y = train.drop(columns=['label']), train['label']
            model.fit(train_x, train_y)
            test = feature[feature['Date'] == unq_date[i+WINDOW_SIZE]]
            test.drop(columns=['Date', 'stock_id'], inplace=True)
            test_x, test_y = test.drop(columns=['label']), test['label']
            prediction = model.predict(test_x)
            pred_dict[unq_date[i+WINDOW_SIZE]] = prediction
            mae.append(mean_absolute_percentage_error(test_y, prediction))
        print(np.mean(mae))
        pred_df = pd.DataFrame(pred_dict, index=pred_dict['stock_id'])
        # tr_feature = train.drop(columns=['Date', 'label', 'stock_id'])
        # ts_feature = test.drop(columns=['Date', 'label', 'stock_id'])
        # regressor = Trainer(args.model_type, cfg[args.model_type])
        # regressor.train(tr_feature, train['label'])
        # prediction = regressor.test(ts_feature)
        # test['pred'] = prediction
        # mae = regressor.eval_metrics(test['label'], prediction)
        # print(mae)

    elif args.model_type.endswith("rk"):
        model = XGBRanker(**cfg[args.model_type])
        # rank_model = Trainer(args.model_type, cfg[args.model_type])
        grouped = train.groupby('Date')
        gkf = GroupKFold(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in gkf.split(train, groups=train['Date']):
            train_data = train.iloc[train_idx]
            val_data = train.iloc[val_idx]

            X_train = train_data.drop(columns=['Date', 'label', 'stock_id'])
            y_train = train_data['label']
            groups_train = train_data.groupby('Date')['stock_id'].count().values
            X_val = val_data.drop(columns=['Date', 'label', 'stock_id'])
            y_val = val_data['label']
            groups_val = val_data.groupby('Date')['stock_id'].count().values
            model.fit(
                X_train, y_train,
                group=groups_train,
                eval_set=[(X_val, y_val)],
                eval_group=[groups_val],
                eval_metric='ndcg@10',
                early_stopping_rounds=200,
                verbose=100
            )

            val_data['pred'] = model.predict(X_val)
            scores = []
            for _, group in val_data.groupby('Date'):
                score = ndcg_score(
                    [group['label']],
                    [group['pred']],
                    k=10
                )
                scores.append(score)

            cv_scores.append(np.mean(score))

        print(f"Average NDCG score: {np.mean(cv_scores)}")

        predictions = model.predict(test.drop(columns=['Date', 'label', 'stock_id']))
        test['pred'] = predictions
        plt.hist(test['pred'])
        plt.show()

    # former = Portfolio(test, org_test)
    # ret = former.calc_merge_ret()
    # former.plot()
