import torch

from torch.utils.data import Dataset

from src.configs import DataConfig
from src.curriculum import Curriculum


# Write tests for this
def get_scm_sampler(conf: DataConfig, **kwargs):
    names_to_classes = {
        "linear": LinearSCMSampler,
    }
    if conf.scm.scm_type in names_to_classes:
        sampler_cls = names_to_classes[conf.scm.scm_type]
        return sampler_cls(conf, **kwargs)
    else:
        raise NotImplementedError


# TODO write tests for this
# TODO add normalizing as a transform func
class SyntheticSCMSampler:
    def __init__(self, conf: DataConfig):
        self.scm_conf = conf.scm
        self.num_samples = conf.num_samples
        self.curriculum = Curriculum(conf.curriculum)
        self.normalize = conf.normalize
        self.data = None

    def generate_data(self, num_samples: int):
        raise NotImplementedError

    def get_data(self, batch_size: int, step_number: int):
        raise NotImplementedError


class LinearSCMSampler(SyntheticSCMSampler):
    def __init__(self, conf: DataConfig):
        super().__init__(conf)
        if conf.scm.t_dim != 1 or conf.scm.y_dim != 1:
            raise NotImplementedError(
                "Only 1-dimensional treatments and outcomes are supported"
            )

        self.data = (
            self.generate_data(self.num_samples) if self.num_samples > 0 else None
        )

    def _noise_sample(self, shape) -> torch.tensor:
        sampler = None

        # choose a random noise type
        rand_idx = torch.randint(len(self.scm_conf.noise_types), (1,))
        sampler = self.scm_conf.noise_types[rand_idx]
        return sampler.sample(shape)

    def generate_data(self, num_scms: int):
        n_points = self.curriculum.get_max_points()
        x = self._noise_sample((num_scms, n_points, self.scm_conf.x_dim))
        e_t = self._noise_sample((num_scms, n_points, self.scm_conf.t_dim))
        e_y = self._noise_sample((num_scms, n_points, self.scm_conf.y_dim))
        w_t = self._noise_sample((num_scms, self.scm_conf.x_dim, self.scm_conf.t_dim))
        w_y = self._noise_sample(
            (num_scms, self.scm_conf.x_dim + self.scm_conf.t_dim, self.scm_conf.y_dim)
        )

        return {"x": x, "e_t": e_t, "e_y": e_y, "w_t": w_t, "w_y": w_y}

    def get_data(self, batch_size: int, step_number: int):
        if self.data is not None:
            batch_idx = step_number % (self.num_samples // batch_size)
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, self.num_samples)
            indices = torch.arange(start, end)
            scm = self.data
        else:
            scm = self.generate_data(batch_size)
            indices = torch.arange(0, batch_size)

        points = self.curriculum.get_n_points(step_number)
        n_dims_trunc = self.curriculum.get_n_dims(step_number)

        x = scm["x"][indices, :points, :]
        x[:, :, n_dims_trunc:] = 0

        e_t = scm["e_t"][indices, :points, :]
        e_y = scm["e_y"][indices, :points, :]
        w_t = scm["w_t"][indices]
        w_y = scm["w_y"][indices]

        treatments_logits_b = torch.einsum("bpx,bxt->bpt", x, w_t) + e_t

        # TODO add confounding factor
        # TODO maybe no bernoulli treatments
        t = torch.bernoulli(torch.sigmoid(treatments_logits_b))

        y = torch.einsum("bpz,bzy->bpy", torch.cat((x, t), dim=2), w_y) + e_y

        # average treatment effect
        ATE = w_y[:, -1, -1]
        if self.normalize:
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(
                dim=1, keepdim=True
            ).clamp_min(1e-5)
            y = (y - y.mean(dim=1, keepdim=True)) / y.std(
                dim=1, keepdim=True
            ).clamp_min(1e-5)

        return ATE, torch.cat((x, t, y), dim=2)


class SyntheticSCMDataset(Dataset):
    def __init__(self, sampler: SyntheticSCMSampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size

    def __getitem__(self, index):
        return self.sampler.get_data(self.batch_size, index)

    def get_current_dim(self, index):
        return self.sampler.curriculum.get_n_dims(index)

    def get_current_points(self, index):
        return self.sampler.curriculum.get_n_points(index)
