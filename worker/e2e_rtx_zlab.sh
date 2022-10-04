#!/bin/bash
#SBATCH --job-name=e2e
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=10:00:00

source ~/.slurmrc

nodename=$(sstat -no "Nodelist" $SLURM_JOB_ID)
echo $nodename

cat $0 | envsubst
echo "--------------------"

python gen_e2e.py $out --model $model --print_int $print --n_shard $n_shard --worker_id $worker
