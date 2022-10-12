#!/bin/bash
#SBATCH --job-name=mt-101
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

python xglm_mt_greedy.py $src_lang $tgt_lang $demo_split $infr_split $out_dir --n_demo $n_demos --model $model --worker_id $worker_id --n_workers $n_workers
