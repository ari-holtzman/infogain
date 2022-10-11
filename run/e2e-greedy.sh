export out_dir=$1
export model=$2
export n_shard=$3
let "top_worker = $n_shard-1"
export print=$4
export script=$5

export log=$(date +'%Y-%m-%d')
mkdir "rec/${log}"
export log_path="rec/${log}/slurm-%j.out"
for worker in `seq -f "%03g" 0 $top_worker` ; do
    export out="${out_dir}/$worker.jsonl"
    sbatch --export=out=$out,model=$model,gen_bs=$gen_bs,ig_bs=$ig_bs,print=$print,n_shard=$n_shard,worker=$worker -o $log_path --parsable $script
done
