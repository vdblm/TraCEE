import os
import yaml
import json

from tqdm import tqdm

import torch

from src.models import build_model, TransformerModel
from src.baselines import get_relevant_baselines
from src.configs import TrainConfig, SCMConfig, MODELS
from src.samplers import get_scm_sampler


def get_model_from_run(run_path, step=-1, only_conf=False) -> (torch.nn.Module, TrainConfig):
    with open(os.path.join(run_path, "config.yaml")) as fp:
        conf = TrainConfig(**yaml.safe_load(fp))
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


def eval_model(
    model: TransformerModel,
    scm_conf: SCMConfig,
    n_points,
    batch_size=64,
    num_eval_examples=1280
):
    assert num_eval_examples % batch_size == 0

    scm_sampler = get_scm_sampler(scm_conf)

    all_metrics = []

    for _ in range(num_eval_examples // batch_size):
        ates, xtys = scm_sampler.sample_xtys(
            n_points=n_points, b_size=batch_size)
        if torch.cuda.is_available() and model.name.split("_")[0] in MODELS:
            device = "cuda"
        else:
            device = "cpu"
        ates = ates.repeat(n_points, 1).T
        pred = model(xtys.to(device)).detach()
        metrics = (pred.cpu() - ates).square()
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)
    results = {}
    results["mean"] = metrics.mean(dim=0).tolist()
    results["std"] = metrics.std(dim=0, unbiased=True).tolist()

    return results


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model = model.cuda().eval()
        all_models = [model]
        if not skip_baselines:
            all_models += get_relevant_baselines(conf.scm.scm_type)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    try:
        with open(save_path) as fp:
            metrics = json.load(fp)
    except Exception:
        metrics = {}

    for model in tqdm(all_models):
        if model.name in metrics and not recompute:
            continue

        metrics[model.name] = eval_model(
            model, scm_conf=conf.scm, n_points=conf.curriculum.points.end, batch_size=conf.batch_size)

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
    return metrics