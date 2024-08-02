"""
Train the model
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    ndcg_score
)
from xgboost import (
    XGBRegressor,
    XGBRanker
)
from lightgbm import (
    LGBMRegressor,
    LGBMRanker
)
from catboost import (
    CatBoostRegressor,
    CatBoostRanker
)

class Trainer:
    """
    Trainer class to train and test the model
    """
    def __init__(self, model_type: str, params: Dict[str, Any]) -> None:
        self.model_type = model_type
        self.params = params
        self.model = self._get_model()

    def _get_model(self) -> Any:
        model_map = {
            "xgbrg": XGBRegressor,
            "xgbrk": XGBRanker,
            "lgbmrg": LGBMRegressor,
            "lgbmrk": LGBMRanker,
            "catbrg": CatBoostRegressor,
            "catbrk": CatBoostRanker
        }
        return model_map[self.model_type](**self.params)

    def train(
        self,
        x_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        grp_train: Optional[np.ndarray] | None = None,
    ) -> None:
        """
        training session
        """
        if self.model_type.endswith("rg"):
            self.model.fit(x_train, y_train)

        elif self.model_type.endswith("rk"):
            self.model.fit(
                x_train,
                y_train,
                qid=grp_train,
            )

    def test(self, x_test: pd.DataFrame) -> pd.Series:
        """
        testing session
        """
        return self.model.predict(x_test)

    def eval_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        evaluate the model
        """
        return mean_absolute_error(y_true, y_pred)
