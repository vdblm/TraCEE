import torch
import torch.nn as nn


def get_relevant_baselines(scm_type: str):
    task_to_baselines = {
        "linear": [
            (LinearDML, {}),
        ],
    }

    models = [model_cls(**kwargs)
              for model_cls, kwargs in task_to_baselines[scm_type]]
    return models

class LinearDML(nn.Module):
    def __init__(self):
        self.name = "DML"
    
    def __call__(self, xtys):
        b_size = xtys.shape[0]
        n_points = xtys.shape[1]
        n_dims = xtys.shape[2] - 2
        preds = []
        for i in range(n_points):
            if i == 0:
                preds.append(torch.zeros(b_size, 1))
                continue
            train_xs, train_ts, train_ys = xtys[:, :i, :n_dims], xtys[:, :i, n_dims:(n_dims+1)], xtys[:, :i, (n_dims+1):]
            wy, _, _, _ = torch.linalg.lstsq(train_xs, train_ys)
            wt, _, _, _ = torch.linalg.lstsq(train_xs, train_ts)
            yhats = train_ys - torch.matmul(train_xs, wy)
            thats = train_ts - torch.matmul(train_xs, wt)
            
            ate, _, _, _ = torch.linalg.lstsq(thats, yhats)
            preds.append(ate.reshape(b_size, 1))
    
        return torch.cat(preds, dim=1)

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
