import os
import uuid
import torch

from clearml import Task

from src.models import build_model
from src.trainer import train
from src.configs import TrainConfig, populate_config
from src.eval import get_run_metrics

import hydra
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: TrainConfig):
    task = Task.init(project_name="TraCEE", task_name="Example")
    
    conf = populate_config(TrainConfig(**OmegaConf.to_object(conf)))

    if conf.test_run:
        conf.curriculum.points.start = conf.curriculum.points.end
        conf.curriculum.dims.start = conf.curriculum.dims.end
        conf.train_steps = 100
    else:
        run_id = conf.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(conf.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        conf.out_dir = out_dir

        OmegaConf.save(conf.dict(), os.path.join(out_dir, "config.yaml"))

    model = build_model(conf)

    train(model, conf)

    if not conf.test_run:
        _ = get_run_metrics(conf.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    main()
