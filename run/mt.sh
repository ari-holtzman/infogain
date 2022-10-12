export lang_file=$1
export model=$2
export demo_split=$3
export infr_split=$4
export n_demos=$5
export out_dir=$6
export n_workers=$7
export script=$8
let "top_worker = $n_workers-1"
export log=$(date +'%Y-%m-%d')
mkdir "rec/${log}"
export log_path="rec/${log}/slurm-%j.out"
while read src_lang ; do
    while read tgt_lang ; do
        if [ $src_lang != $tgt_lang ] ; then
            for worker_id in `seq 0 $top_worker` ; do
                sbatch --export=src_lang=$src_lang,tgt_lang=$tgt_lang,demo_split=$demo_split,infr_split=$infr_split,out_dir=$out_dir,n_demos=$n_demos,model=$model,worker_id=$worker_id,n_workers=$n_workers -o $log_path $script
            done
        fi
    done <$lang_file
done <$lang_file
