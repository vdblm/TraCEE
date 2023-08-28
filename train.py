import torch
import hydra
import wandb

import numpy as np
import random

from src.models import build_model
from src.trainer import train
from src.configs import TraCEEConfig
from src.eval import get_run_metrics
from src.utils import init_run_dir

from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)

# TODO might use mark_preempting
# TODO handle multiple runs per agent vs re-running the same run for preemption
# TODO handle removing a run
# TODO hyperparam sweep over n_embed, n_layer, n_heads
# TODO make sure the training procedure actually learns the algorithm


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: TraCEEConfig):
    conf = TraCEEConfig(**OmegaConf.to_object(conf))

    # reproducibility
    torch.manual_seed(conf.seed)

    np.random.seed(conf.seed)
    random.seed(conf.seed)

    if conf.test_run:
        conf.curriculum.points.start = conf.curriculum.points.end
        conf.curriculum.dims.start = conf.curriculum.dims.end
        conf.train.train_steps = 100
    else:
        conf = init_run_dir(conf)

        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.dict(),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )

    model = build_model(conf)

    train(model, conf)

    if not conf.test_run:
        _ = get_run_metrics(conf.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    main()
