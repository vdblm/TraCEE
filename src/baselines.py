import torch

from torch import nn
from sklearn.linear_model import LogisticRegression


# TODO write tests for baselines


def get_relevant_baselines(scm_type: str):
    task_to_baselines = {
        "linear": [
            (LinearDML, {}),
            (GEstimationModel, {}),
            (PropensityEstimationModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[scm_type]]
    return models

# TODO use doubleml package to get ATEs and CIs
class LinearDML(nn.Module):
    def __init__(self):
        self.name = "DML"

    def __call__(self, xtys):
        b_size = xtys.shape[0]
        n_points = xtys.shape[1]
        x_dims = xtys.shape[2] - 2
        means = []
        stds = []
        for i in range(n_points):
            if i == 0:
                means.append(torch.zeros(b_size, 1))
                stds.append(torch.zeros(b_size, 1))
                continue

            train_xs, train_ts, train_ys = (
                xtys[:, : (i + 1), :x_dims],
                xtys[:, : (i + 1), x_dims : (x_dims + 1)],
                xtys[:, : (i + 1), (x_dims + 1) :],
            )
            wy, _, _, _ = torch.linalg.lstsq(train_xs, train_ys)
            wt, _, _, _ = torch.linalg.lstsq(train_xs, train_ts)
            yhats = train_ys - torch.matmul(train_xs, wy)
            thats = train_ts - torch.matmul(train_xs, wt)
            ate, _, _, _ = torch.linalg.lstsq(thats, yhats)
            means.append(ate.reshape(b_size, 1))
            stds.append(torch.zeros((b_size, 1)))

        return torch.cat(means, dim=1), 2 * torch.log(torch.cat(stds, dim=1))


class GEstimationModel:
    def __init__(self):
        self.name = "G-estimation"

    def __call__(self, xtys):
        b_size, n_points, total_dims = xtys.shape
        x_dims = total_dims - 2
        estimated_ates = []
        estimated_cis = []

        for i in range(n_points):
            train_xs, train_ts, train_ys = (
                xtys[:, :i + 1, :x_dims].reshape(-1, x_dims),
                xtys[:, :i + 1, x_dims].reshape(-1, 1),
                xtys[:, :i + 1, x_dims + 1].reshape(-1, 1),
            )

            # Linear regression using PyTorch
            ones = torch.ones(train_ts.shape[0], 1)
            treated_xs = torch.hstack((ones, train_xs))
            untreated_xs = torch.hstack((torch.zeros_like(ones), train_xs))

            wy, _ = torch.linalg.lstsq(treated_xs, train_ys).solution[:x_dims+1]
            treated_outcome = torch.matmul(treated_xs, wy)

            wy, _ = torch.linalg.lstsq(untreated_xs, train_ys).solution[:x_dims+1]
            untreated_outcome = torch.matmul(untreated_xs, wy)

            # Calculate ATE and its standard error for this iteration
            ate = torch.mean(treated_outcome - untreated_outcome)
            se_ate = torch.std(treated_outcome - untreated_outcome) / torch.sqrt(torch.tensor(train_ts.shape[0]))

            estimated_ates.append(ate)
            ci = 1.96 * se_ate
            estimated_cis.append(ci)

        return torch.tensor(estimated_ates).reshape(b_size, -1), torch.tensor(estimated_cis).reshape(b_size, -1)


class PropensityEstimationModel:
    def __init__(self):
        self.name = "Propensity"

    def __call__(self, xtys):
        # Runs Propensity estimation
        # Returns ATE and CI
        b_size, n_points, total_dims = xtys.shape
        x_dims = total_dims - 2
        estimated_ates = []
        estimated_cis = []

        for i in range(n_points):
            train_xs, train_ts, train_ys = (
                xtys[:, :i + 1, :x_dims].reshape(-1, x_dims),
                xtys[:, :i + 1, x_dims].reshape(-1, 1),
                xtys[:, :i + 1, x_dims + 1].reshape(-1, 1),
            )

            # Fit a logistic regression model using sklearn
            model = LogisticRegression()
            model.fit(train_xs.numpy(), train_ts.numpy().ravel())

            # Predict probabilities of treatment using sklearn but convert to PyTorch tensor
            propensity_scores = torch.tensor(model.predict_proba(train_xs.numpy())[:, 1])

            # Calculate weights using PyTorch
            weights = torch.where(train_ts == 1,
                                  1 / propensity_scores,
                                  1 / (1 - propensity_scores))

            # Normalize weights
            weights /= torch.mean(weights)

            # Calculate ATE and its standard error for this iteration using PyTorch
            treated_outcome = torch.mean(weights * train_ys * train_ts)
            untreated_outcome = torch.mean(weights * train_ys * (1 - train_ts))
            ate = treated_outcome - untreated_outcome
            se_ate = torch.std(weights * (train_ys - ate) * train_ts) / torch.sqrt(torch.tensor(len(train_ts)))

            estimated_ates.append(ate)
            ci = 1.96 * se_ate  # 95% CI
            estimated_cis.append(ci)

        return torch.tensor(estimated_ates).reshape(b_size, -1), torch.tensor(estimated_cis).reshape(b_size, -1)
