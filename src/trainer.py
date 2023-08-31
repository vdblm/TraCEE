import os
from tqdm import tqdm
import torch

import torch.nn as nn

from src.configs import TraCEEConfig
from src.curriculum import Curriculum
from src.samplers import get_scm_sampler, SyntheticSCMDataset

import wandb


def train_step(model, xs, ys, optimizer):
    optimizer.zero_grad()
    mean, log_var = model(xs)
    loss_func = nn.GaussianNLLLoss()
    loss = loss_func(mean, ys, torch.exp(log_var))
    loss.backward()
    optimizer.step()
    return loss.detach().item(), mean.detach()


def eval_step(model, xs, ys):
    mean, log_var = model(xs)
    loss_func = nn.GaussianNLLLoss()
    loss = loss_func(mean, ys, torch.exp(log_var))
    return loss.detach().item(), mean.detach()


def pointwise_loss(output, ATEs) -> wandb.Table:
    point_wise_tags = list(range(output.shape[1]))
    point_wise_loss = (output - ATEs.cuda()).square().mean(dim=0)
    df = wandb.Table(
        data=list(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
        columns=["samples", "loss"],
    )

    return df


# TODO clean the validation/train steps + logging
def train(model: nn.Module, conf: TraCEEConfig):
    model.cuda()
    model.train()

    # initialize optimizer
    # TODO make this configurable
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.trainer.learning_rate)

    starting_step = 0

    # resume if the training was interrupted
    state_path = os.path.join(conf.out_dir, "state.pt")
    if conf.wandb.resume:
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

    train_dataset = SyntheticSCMDataset(
        get_scm_sampler(conf=conf.train_data), conf.trainer.batch_size
    )
    val_dataset = SyntheticSCMDataset(
        get_scm_sampler(conf=conf.val_data), conf.trainer.batch_size
    )

    pbar = tqdm(range(starting_step, conf.trainer.train_steps))

    for i in pbar:
        # train
        model.train()
        ATEs, xtys = train_dataset[i]
        n_points = train_dataset.get_current_points(i)
        n_dims = train_dataset.get_current_dim(i)
        ATEs = ATEs.repeat(n_points, 1).T

        loss, output = train_step(model, xtys.cuda(), ATEs.cuda(), optimizer)

        # LOGGING
        if i % conf.wandb.log_every_steps == 0 and not conf.test_run:
            wandb.log(
                {"overall/loss": loss, "n_points": n_points, "n_dims": n_dims},
                step=i,
            )

            # validate
            model.eval()
            val_ATEs, val_xtys = val_dataset[i]

            val_ATEs = val_ATEs.repeat(n_points, 1).T
            val_loss, val_output = eval_step(model, val_xtys.cuda(), val_ATEs.cuda())
            wandb.log({"overall/val-loss": val_loss}, step=i)

        pbar.set_description(f"loss {loss}")
        if i % conf.save_every_steps == 0 and not conf.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            conf.keep_every_steps > 0
            and i > 0
            and i % conf.keep_every_steps == 0
            and not conf.test_run
        ):
            torch.save(model.state_dict(), os.path.join(conf.out_dir, f"model_{i}.pt"))

            # TODO fix this
            val_df = pointwise_loss(val_output, val_ATEs)
            wandb.log(
                {
                    "pointwise/val-loss": wandb.plot.line(
                        val_df, "samples", "loss", title="val"
                    )
                },
                step=i,
            )

            # pointwise data
            train_df = pointwise_loss(output, ATEs)
            wandb.log(
                {
                    "pointwise/loss": wandb.plot.line(
                        train_df, "samples", "loss", title="train"
                    )
                },
                step=i,
            )
