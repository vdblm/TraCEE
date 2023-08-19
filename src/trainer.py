import os
from tqdm import tqdm
import torch

import torch.nn as nn

from src.configs import TraCEEConfig
from src.curriculum import Curriculum
from src.samplers import get_scm_sampler

import wandb


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def train(model: nn.Module, conf: TraCEEConfig):
    model.cuda()
    model.train()

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=conf.train.learning_rate)

    curriculum = Curriculum(conf.curriculum)

    starting_step = 0

    # resume if the training was interrupted
    state_path = os.path.join(conf.out_dir, "state.pt")
    if conf.wandb.resume:
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

        # update the curriculum
        for i in range(state["train_step"] + 1):
            curriculum.update()

    bsize = conf.train.batch_size
    scm_sampler = get_scm_sampler(conf=conf.scm)
    pbar = tqdm(range(starting_step, conf.train.train_steps))

    for i in pbar:

        ATEs, xtys = scm_sampler.sample_xtys(n_points=curriculum.n_points, b_size=bsize,
                                             n_dims_trunc=curriculum.n_dims_truncated)
        ATEs = ATEs.repeat(curriculum.n_points, 1).T

        loss_func = nn.MSELoss()

        loss, output = train_step(
            model, xtys.cuda(), ATEs.cuda(), optimizer, loss_func)

        # LOGGING
        if i % conf.wandb.log_every_steps == 0 and not conf.test_run:
            wandb.log({"overall/loss": loss, "n_points": curriculum.n_points,
                      "n_dims": curriculum.n_dims_truncated}, step=i)

            # pointwise data
            point_wise_tags = list(range(curriculum.n_points))
            point_wise_loss = (output - ATEs.cuda()).square().mean(dim=0)
            df = wandb.Table(
                data=list(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
                columns=["samples", "loss"])
            wandb.log(
                {"pointwise/loss": wandb.plot.line(df, "samples", "loss")}, step=i)

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % conf.save_every_steps == 0 and not conf.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            conf.keep_every_steps > 0 and i > 0
            and i % conf.keep_every_steps == 0
            and not conf.test_run
        ):
            torch.save(model.state_dict(), os.path.join(
                conf.out_dir, f"model_{i}.pt"))
