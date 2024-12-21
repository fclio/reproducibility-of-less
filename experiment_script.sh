# repository & output directory locations. change if not running on Snellius
export REPODIR="/home/scur2847/ir2-less-data"
export OUTDIR="/scratch-shared/ir2-less"

# arguments
LESS_PIPELINE_SECTION=$1
MODEL=$2
SEED=$3
CMD=$4

# default argument values
if [ -z "$LESS_PIPELINE_SECTION" ]
then
    LESS_PIPELINE_SECTION=unknown
fi

if [ -z "$MODEL" ]
then
    MODEL=llama2-7b
fi

if [ -z "$SEED" ]
then
    SEED=5
fi

if [ -z "$CMD" ]
then
    CMD=sbatch
fi

PERCENTAGE=0.05

echo $LESS_PIPELINE_SECTION $MODEL $SEED $CMD

if [[ "$1" == "install" ]]; then
    ${CMD} experiment_scripts/install_LESS.sh 
elif [[ "$1" == "warmup" ]]; then
    ${CMD} experiment_scripts/warmup.sh ${MODEL} ${SEED} - $REPODIR $OUTDIR
elif [[ "$1" == "warmup_ir" ]]; then
    ${CMD} experiment_scripts/warmup.sh ${MODEL} ${SEED} IR $REPODIR $OUTDIR
elif [[ "$1" == "datastore" ]]; then
    if [[ "$MODEL" == "llama2-7b" ]]; then
        for task in "cot" "dolly" "flan_v2" "oasst1"
        do
            for checkpoint in "422" "845" "1268" "1688"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} - $REPODIR $OUTDIR
            done
        done

        for task in "mmlu" "tydiqa" "bbh"
        do
            for checkpoint in "422" "845" "1268" "1688"
            do 
                ${CMD} experiment_scripts/build_eval_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} $REPODIR $OUTDIR
            done
        done
    elif [[ "$MODEL" == "llama2-13b" ]]; then
        for task in "cot" "dolly" "flan_v2" "oasst1"
        do
            for checkpoint in "211" "422" "634" "844"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} - $REPODIR $OUTDIR
            done
        done

        for task in "mmlu" "tydiqa" "bbh"
        do
            for checkpoint in "211" "422" "634" "844"
            do 
                ${CMD} experiment_scripts/build_eval_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} $REPODIR $OUTDIR
            done
        done
    fi
elif [[ "$1" == "datastore_ir" ]]; then
    if [[ "$MODEL" == "llama2-7b" ]]; then
        for task in "first" "nfcorpus" "scifact" "vihealthqa" "fiqa"
        do
            for checkpoint in "31" "62" "93" "124"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} IR $REPODIR $OUTDIR
            done
        done
    elif [[ "$MODEL" == "llama2-13b" ]]; then
        for task in "first" "nfcorpus" "scifact" "vihealthqa" "fiqa"
        do
            for checkpoint in "31" "62" "93" "124"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} IR $REPODIR $OUTDIR
            done
        done
    fi
elif [[ "$1" == "matching" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/matching.sh $task ${MODEL} ${SEED} 1234 $REPODIR $OUTDIR
    done
elif [[ "$1" == "matching_ir" ]]; then
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/matching_ir.sh $task ${MODEL} ${SEED} 1234 $REPODIR $OUTDIR
    done
elif [[ "$1" == "matching_baseline" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/matching_random.sh $task ${MODEL} ${SEED} - $REPODIR $OUTDIR
        ${CMD} experiment_scripts/matching_bm25.sh $task ${MODEL} ${SEED} - $REPODIR $OUTDIR
    done
elif [[ "$1" == "matching_baseline_ir" ]]; then
    #TODO
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/matching_random.sh $task ${MODEL} ${SEED} IR $REPODIR $OUTDIR
        ${CMD} experiment_scripts/matching_bm25.sh $task ${MODEL} ${SEED} IR $REPODIR $OUTDIR
    done
elif [[ "$1" == "matching_ablation" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
        do
            ${CMD} experiment_scripts/matching.sh $task ${MODEL} ${SEED} $ckptidxs $REPODIR $OUTDIR
        done
    done
elif [[ "$1" == "finetune" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 - $REPODIR $OUTDIR
    done
elif [[ "$1" == "finetune_ir" ]]; then
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR $REPODIR $OUTDIR
    done
elif [[ "$1" == "finetune_transfer" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 T $REPODIR $OUTDIR
    done
elif [[ "$1" == "finetune_baseline" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 BM25 $REPODIR $OUTDIR
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 random $REPODIR $OUTDIR
    done
elif [[ "$1" == "finetune_baseline_ir" ]]; then
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR_BM25 $REPODIR $OUTDIR
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR_random $REPODIR $OUTDIR
    done
elif [[ "$1" == "finetune_ablation" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
        do
            ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} $ckptidxs - $REPODIR $OUTDIR
        done
    done
elif [[ "$1" == "eval" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-1234 $REPODIR $OUTDIR
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-1234 $REPODIR $OUTDIR
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-1234 $REPODIR $OUTDIR
elif [[ "$1" == "eval_ir" ]]; then
    cd evaluation
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-1234 $task $REPODIR $OUTDIR
    done
elif [[ "$1" == "eval_transfer" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-1234-T $REPODIR $OUTDIR
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-1234-T $REPODIR $OUTDIR
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-1234-T $REPODIR $OUTDIR
elif [[ "$1" == "eval_baseline" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-BM25 $REPODIR $OUTDIR
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-BM25 $REPODIR $OUTDIR
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-BM25 $REPODIR $OUTDIR
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-random $REPODIR $OUTDIR
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-random $REPODIR $OUTDIR
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-random $REPODIR $OUTDIR
elif [[ "$1" == "eval_baseline_ir" ]]; then
    cd evaluation
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-random $task $REPODIR $OUTDIR
        ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-BM25 $task $REPODIR $OUTDIR
    done
elif [[ "$1" == "eval_ablation" ]]; then
    cd evaluation
    for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
    do
        ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-$ckptidxs $REPODIR $OUTDIR
        ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-$ckptidxs $REPODIR $OUTDIR
        ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-$ckptidxs $REPODIR $OUTDIR
    done
else
    echo "Unknown LESS pipeline section. Check the README for information on how to use the script."
fi

