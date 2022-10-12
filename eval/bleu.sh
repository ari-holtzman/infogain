data_dir=$1
ref_dir=$2
comp_dir=$3
spm_dir=$4
bleu_dir=$5
lang_file=$6
while read src_lang ; do
    while read tgt_lang ; do
        echo $tgt_lang
        if [ $src_lang != $tgt_lang ] ; then
            name=$src_lang-$tgt_lang
            comp_path=$comp_dir/$name.hypo
            spm_path=$spm_dir/$name.hypo.spm
            ref_path=$ref_dir/$tgt_lang.ref.spm
            bleu_path=$bleu_dir/$name.bleu
            paste -d "\n" $data_dir/$src_lang-$tgt_lang* > $comp_path
            python ../fairseq/scripts/spm_encode.py --model ../flores/flores200sacrebleuspm --output_format=piece --inputs $comp_path --outputs $spm_path
            cat $spm_path | sacrebleu $ref_path > $bleu_path
        fi
    done <$lang_file
done <$lang_file
