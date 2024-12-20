LESS_PIPELINE_SECTION=$1
MODEL=$2
SEED=$3
CMD=$4

# default values
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
    ${CMD} experiment_scripts/warmup.sh ${MODEL} ${SEED}
elif [[ "$1" == "warmup_ir" ]]; then
    ${CMD} experiment_scripts/warmup.sh ${MODEL} ${SEED} IR
elif [[ "$1" == "datastore" ]]; then
    if [[ "$MODEL" == "llama2-7b" ]]; then
        for task in "cot" "dolly" "flan_v2" "oasst1"
        do
            for checkpoint in "422" "845" "1268" "1688"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED}
            done
        done

        for task in "mmlu" "tydiqa" "bbh"
        do
            for checkpoint in "422" "845" "1268" "1688"
            do 
                ${CMD} experiment_scripts/build_eval_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED}
            done
        done
    elif [[ "$MODEL" == "llama2-13b" ]]; then
        for task in "cot" "dolly" "flan_v2" "oasst1"
        do
            for checkpoint in "211" "422" "634" "844"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED}
            done
        done

        for task in "mmlu" "tydiqa" "bbh"
        do
            for checkpoint in "211" "422" "634" "844"
            do 
                ${CMD} experiment_scripts/build_eval_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED}
            done
        done
    fi
elif [[ "$1" == "datastore_ir" ]]; then
    if [[ "$MODEL" == "llama2-7b" ]]; then
        for task in "first" "nfcorpus" "scifact" "vihealthqa" "fiqa"
        do
            for checkpoint in "31" "62" "93" "124"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} IR
            done
        done
    elif [[ "$MODEL" == "llama2-13b" ]]; then
        for task in "first" "nfcorpus" "scifact" "vihealthqa" "fiqa"
        do
            for checkpoint in "31" "62" "93" "124"
            do 
                ${CMD} experiment_scripts/build_train_grad_datastore.sh $task $checkpoint ${MODEL} ${SEED} IR
            done
        done
    fi
elif [[ "$1" == "matching" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/matching.sh $task ${MODEL} ${SEED} 1234
    done
elif [[ "$1" == "matching_ir" ]]; then
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/matching_ir.sh $task ${MODEL} ${SEED} 1234
    done
elif [[ "$1" == "matching_baseline" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/matching_random.sh $task ${MODEL} ${SEED}
        ${CMD} experiment_scripts/matching_bm25.sh $task ${MODEL} ${SEED}
    done
elif [[ "$1" == "matching_baseline_ir" ]]; then
    #TODO
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/matching_random.sh $task ${MODEL} ${SEED} IR
        ${CMD} experiment_scripts/matching_bm25.sh $task ${MODEL} ${SEED} IR
    done
elif [[ "$1" == "matching_ablation" ]]; then
    # for task in "mmlu" "tydiqa" "bbh"
    for task in "mmlu"
    do
        for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
        do
            ${CMD} experiment_scripts/matching.sh $task ${MODEL} ${SEED} $ckptidxs
        done
    done
elif [[ "$1" == "finetune" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234
    done
elif [[ "$1" == "finetune_ir" ]]; then
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR
    done
elif [[ "$1" == "finetune_transfer" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 T
    done
elif [[ "$1" == "finetune_baseline" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 BM25
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 random
    done
elif [[ "$1" == "finetune_baseline_ir" ]]; then
    #TODO
    # for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    for task in "scifact"
    do
        # ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR_BM25
        ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} 1234 IR_random
    done
elif [[ "$1" == "finetune_ablation" ]]; then
    for task in "mmlu" "tydiqa" "bbh"
    do
        for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
        do
            ${CMD} experiment_scripts/training.sh $task ${MODEL} ${SEED} $ckptidxs
        done
    done
elif [[ "$1" == "eval" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-1234
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-1234
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-1234
elif [[ "$1" == "eval_ir" ]]; then
    cd evaluation
    for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    do
        ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-1234 $task
    done
elif [[ "$1" == "eval_transfer" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-1234-T
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-1234-T
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-1234-T
elif [[ "$1" == "eval_baseline" ]]; then
    cd evaluation
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-BM25
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-BM25
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-BM25
    ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-random
    ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-random
    ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-random
elif [[ "$1" == "eval_baseline_ir" ]]; then
    cd evaluation
    # for task in "nfcorpus" "scifact" "vihealthqa" "fiqa"
    for task in "scifact"
    do
        ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-random $task
        # ${CMD} eval_ranking_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-$task-BM25 $task
    done
elif [[ "$1" == "eval_ablation" ]]; then
    cd evaluation
    for ckptidxs in "1" "2" "3" "4" "12" "34" "23" "14"
    do
        ${CMD} eval_mmlu_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-mmlu-$ckptidxs
        ${CMD} eval_bbh_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-bbh-$ckptidxs
        ${CMD} eval_tydiqa_after_finetuning.sh ${MODEL}-less-p${PERCENTAGE}-lora-seed${SEED}-tydiqa-$ckptidxs
    done
else
    echo "Unknown LESS pipeline section. Check the README for information on how to use the script."
fi

