# Transformers for Causal Effect Estimation

Run the following code to generate the sweep id:

 `wandb sweep sweeps/slurm_sweep.yaml -p $project`

 which generates `$sweep_id`. Then, run the following code:

 `sbatch slurm/normal_launch.slrm  wandb agent --count 1 $entity/$project/$sweep_id`

 Run all the commands from this directory.
