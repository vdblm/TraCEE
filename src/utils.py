
from src.configs import TraCEEConfig

from random_word import RandomWords
import os
import yaml

import wandb

from omegaconf import OmegaConf


def init_run_dir(conf: TraCEEConfig) -> TraCEEConfig:
    # Handle preemption and resume
    run_name = conf.wandb.run_name
    resume = True
    if run_name is None:
        r = RandomWords()
        w1, w2 = r.get_random_word(), r.get_random_word()
        run_name = f"{w1}_{w2}"

    out_dir = os.path.join(conf.out_dir, run_name)

    config_yaml = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_yaml):
        with open(config_yaml) as fp:
          old_conf = TraCEEConfig(**yaml.safe_load(fp))
        run_id = old_conf.wandb.run_id
    else:
      run_id = wandb.util.generate_id()
      resume = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        resume = False
    elif not os.path.exists(os.path.join(out_dir, "state.pt")):
      resume = False

    conf.out_dir = out_dir
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    OmegaConf.save(conf.dict(), os.path.join(out_dir, "config.yaml"))

    return conf
