import os
import json

from tqdm import tqdm

import torch

from src.models import TransformerModel
from src.baselines import get_relevant_baselines
from src.configs import SCMConfig, MODELS
from src.samplers import get_scm_sampler
from src.utils import get_model_from_run


def eval_model(
    model: TransformerModel,
    scm_conf: SCMConfig,
    n_points,
    batch_size=64,
    num_eval_examples=64,  # TODO increase
):
    assert num_eval_examples % batch_size == 0

    scm_sampler = get_scm_sampler(scm_conf)

    mse = []
    coverage = []

    for _ in range(num_eval_examples // batch_size):
        ates, xtys = scm_sampler.sample_xtys(n_points=n_points, b_size=batch_size)
        if torch.cuda.is_available() and model.name.split("_")[0] in MODELS:
            device = "cuda"
        else:
            device = "cpu"
        ates = ates.repeat(n_points, 1).T
        pred, log_var = model(xtys.to(device))
        pred = pred.detach().cpu()
        log_var = log_var.detach().cpu()
        mse.append((pred - ates).square())

        renge = 1.96 * torch.exp(0.5 * log_var)
        coverage.append((ates >= pred - renge) & (ates <= pred + renge))

    mse = torch.cat(mse, dim=0)
    coverage = torch.cat(coverage, dim=0)
    results = {}
    results["mse"] = mse.mean(dim=0).tolist()
    results["error_std"] = mse.std(dim=0, unbiased=True).tolist()
    results["coverage"] = coverage.float().mean(dim=0).tolist()
    results["coverage_std"] = coverage.float().std(dim=0, unbiased=True).tolist()

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
            model,
            scm_conf=conf.scm,
            n_points=conf.curriculum.points.end,
            batch_size=conf.train.batch_size,
        )

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
    return metrics
