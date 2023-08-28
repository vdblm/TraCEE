import os
import yaml

import torch
import wandb

import pandas as pd

from src.configs import TraCEEConfig
from src.models import build_model

from random_word import RandomWords
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


def get_model_from_run(
    run_path, step=-1, only_conf=False
) -> (torch.nn.Module, TraCEEConfig):
    with open(os.path.join(run_path, "config.yaml")) as fp:
        conf = TraCEEConfig(**yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = build_model(conf)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


def conf_to_model_name(conf: TraCEEConfig):
    return {
        (3, 2): "Transformer-xs",
        (6, 4): "Transformer-small",
        (12, 8): "Transformer",
    }[(conf.model.n_layer, conf.model.n_head)]


def read_run_dir(run_dir):
    all_runs = {}
    for run_id in os.listdir(run_dir):
        run_path = os.path.join(run_dir, run_id)
        try:
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["model"] = conf_to_model_name(conf)
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.run_name
            params["noise_types"] = conf.scm.noise_types

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)
        except:
            continue

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df
