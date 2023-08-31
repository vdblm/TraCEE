import os
import json

from tqdm import tqdm

import torch

from src.baselines import get_relevant_baselines
from src.configs import DataConfig, SCMConfig, MODELS
from src.curriculum import Curriculum
from src.samplers import get_scm_sampler, SyntheticSCMDataset
from src.utils import get_model_from_run


def eval_model(
    model,
    scm_conf: SCMConfig,
    n_points,
    batch_size=64,
    num_eval_examples=1280,
):
    assert num_eval_examples % batch_size == 0
    curriculum_conf = Curriculum.get_fixed_curriculum(n_points, scm_conf.x_dim)
    data_conf = DataConfig(
        scm=scm_conf, curriculum=curriculum_conf, num_samples=num_eval_examples
    )
    dataset = SyntheticSCMDataset(get_scm_sampler(data_conf), batch_size=batch_size)

    mse = []
    coverage = []

    for i in range(num_eval_examples // batch_size):
        ates, xtys = dataset[i]
        if torch.cuda.is_available() and model.name.split("_")[0] in MODELS:
            model.eval().cuda()
            xtys = xtys.to("cuda")

        ates = ates.repeat(n_points, 1).T
        pred, log_var = model(xtys)
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
    run_path, step=-1, skip_model_load=False, skip_baselines=False, recompute=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        all_models = [model]
        if not skip_baselines:
            all_models += get_relevant_baselines(conf.val_data.scm.scm_type)

    if step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

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
            scm_conf=conf.val_data.scm,
            n_points=conf.val_data.curriculum.points.end,
            batch_size=conf.trainer.batch_size,
        )

    with open(save_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics
