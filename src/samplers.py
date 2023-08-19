import torch
from typing import Tuple

import torch.distributions as dist
from src.configs import SCMConfig, NOISE_TYPES


def get_scm_sampler(conf: SCMConfig, **kwargs):
    names_to_classes = {
        "linear": LinearSCMSampler,
    }
    if conf.scm_type in names_to_classes:
        sampler_cls = names_to_classes[conf.scm_type]
        return sampler_cls(conf, **kwargs)
    else:
        raise NotImplementedError


class LinearSCMSampler:
    def __init__(self, conf: SCMConfig):
        if conf.t_dim != 1 or conf.y_dim != 1:
            raise NotImplementedError(
                "Only 1-dimensional treatments and outcomes are supported")
        self.n_dims = conf.x_dim
        self.noise_type = conf.noise_types

        if isinstance(self.noise_type, str):
            self.noise_type = [self.noise_type]

    def noise_sample(self, shape) -> torch.tensor:
        sampler = None

        # choose a random noise type
        noise_type = self.noise_type[torch.randint(len(self.noise_type), (1,))]
        if noise_type == "uniform":
            sampler = dist.Uniform(torch.zeros(shape), torch.ones(shape))
        elif noise_type == "gaussian":
            sampler = dist.Normal(torch.zeros(shape), torch.ones(shape))
        elif noise_type == "laplace":
            sampler = dist.Laplace(torch.zeros(shape), torch.ones(shape))
        elif noise_type == "exponential":
            sampler = dist.Exponential(torch.ones(shape))
        elif noise_type == "log-normal":
            sampler = dist.LogNormal(torch.zeros(shape), torch.ones(shape))
        else:
            raise NotImplementedError

        return sampler.sample()

    def _add_const(self, shape):
        const_range = 1  # TODO make this a parameter
        return 2 * const_range * torch.rand(shape) - const_range

    # TODO add seeds + maybe non-identifiable
    def sample_xtys(self, n_points, b_size, n_dims_trunc=None) -> Tuple[torch.tensor, torch.tensor]:

        covariates_b = self.noise_sample((b_size, n_points, self.n_dims)) + self._add_const(
            (b_size, n_points, self.n_dims))  # dim: b_size x n_points x n_dims

        w_T = self.noise_sample((b_size, self.n_dims)) + \
            self._add_const((b_size, self.n_dims))

        treatments_probs_b = torch.sigmoid(torch.einsum(
            'bnp,bp->bn', covariates_b, w_T) + self.noise_sample((b_size, n_points)))

        treatments_b = torch.bernoulli(
            treatments_probs_b).reshape(b_size, n_points, 1)

        w_Y = self.noise_sample((b_size, self.n_dims + 1)) + \
            self._add_const((b_size, self.n_dims + 1))

        outcomes_b = torch.einsum('bnp,bp->bn', torch.cat(
            (covariates_b, treatments_b), dim=2), w_Y) + self.noise_sample((b_size, n_points))

        outcomes_b = outcomes_b.reshape(b_size, n_points, 1)

        # average treatment effect
        ATE = w_Y[:, -1]

        if n_dims_trunc is not None:
            covariates_b[:, :, n_dims_trunc:] = 0

        # concat covariates_b, treatments_b, outcomes_b
        return ATE, torch.cat((covariates_b, treatments_b, outcomes_b), dim=2)
