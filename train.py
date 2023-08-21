import torch

import wandb

from src.models import build_model
from src.trainer import train
from src.configs import TraCEEConfig
from src.eval import get_run_metrics
from src.utils import init_run_dir

from pprint import pprint

import hydra
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)

# TODO seed

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: TraCEEConfig):
    conf = TraCEEConfig(**OmegaConf.to_object(conf))

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
            settings=wandb.Settings(start_method="thread")
        )

    model = build_model(conf)

    train(model, conf)

    if not conf.test_run:
        _ = get_run_metrics(conf.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    main()
