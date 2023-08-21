#!/bin/bash
ln -sfn /checkpoint/${USER}/* $PWD/output/runs

touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

. $HOME/pytorch-2-env

python train.py wandb.run_name=${SLURM_JOB_ID} "$@"  # Since list can be in quotes, we need to pass "$@" instead of $@
