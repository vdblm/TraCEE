import torch
import doubleml as dml
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from torch import nn


def get_relevant_baselines(scm_type: str):
    task_to_baselines = {
        "linear": [
            (LinearDML, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[scm_type]]
    return models


class LinearDML(nn.Module):
    def __init__(self):
        self.name = "DML"
        # TODO make this faster

    def dml(self, xtys):
        """Runs DML on the one data sample

        Args:
            xtys (torch.tensor): shape (n_points, x_dims + 2)

        Returns:
            mean, log_var: shape (2,)
        """
        x_dims = xtys.shape[1] - 2
        try:
            data_df = pd.DataFrame(
                xtys, columns=[f"x{i}" for i in range(x_dims)] + ["t", "y"]
            )
            data_dml = dml.DoubleMLData(
                data_df, y_col="y", d_cols="t", x_cols=[f"x{i}" for i in range(x_dims)]
            )

            dml_plr = dml.DoubleMLPLR(
                data_dml,
                ml_l=LinearRegression(),
                ml_m=LogisticRegression(),
                n_folds=min(5, data_df.shape[0]),
            )

            dml_plr.fit(store_predictions=True)
            res = torch.Tensor([dml_plr.coef[0], dml_plr.se[0]])
        except:
            train_xs, train_ts, train_ys = (
                xtys[:, :x_dims],
                xtys[:, x_dims : (x_dims + 1)],
                xtys[:, (x_dims + 1) :],
            )
            wy, _, _, _ = torch.linalg.lstsq(train_xs, train_ys)
            wt, _, _, _ = torch.linalg.lstsq(train_xs, train_ts)
            yhats = train_ys - torch.matmul(train_xs, wy)
            thats = train_ts - torch.matmul(train_xs, wt)
            ate, _, _, _ = torch.linalg.lstsq(thats, yhats)
            std = 0
            res = torch.Tensor([ate, std])
        return res.reshape(1, 2)

    def __call__(self, xtys):
        b_size = xtys.shape[0]
        n_points = xtys.shape[1]
        means = []
        stds = []
        for i in range(n_points):
            if i == 0:
                means.append(torch.zeros(b_size, 1))
                stds.append(torch.zeros(b_size, 1))
                continue
            # shape: (b_size, 2)
            res = torch.cat(
                [self.dml(xtys[j, : i + 1, :]) for j in range(b_size)], dim=0
            )
            means.append(res[:, 0].reshape(b_size, 1))
            stds.append(res[:, 1].reshape(b_size, 1))

        return torch.cat(means, dim=1), 2 * torch.log(torch.cat(stds, dim=1))


class GEstimationModel:
    def __init__(self):
        pass

    def __call__(self, xtys, ATE):
        # Runs G-estimation
        pass


class PropensityEstimationModel:
    def __init__(self):
        pass

    def __call__(self, xtys, ATE):
        # Runs Propensity estimation
        pass
