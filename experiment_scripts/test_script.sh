#!/bin/bash

TRAINING_DATA_NAME=$1
CKPT=$2
DATA_SEED=$3
CHECKPOINTS=$4

foo=${CHECKPOINTS}
CKPTS=""
for (( i=0; i<${#foo}; i++ )); do
    echo "${foo:$i:1}"
    if [[ "${foo:$i:1}" == "1" ]]; then
        CKPTS+=" 422"
    elif [[ "${foo:$i:1}" == "2" ]]; then
        CKPTS+=" 845"
    elif [[ "${foo:$i:1}" == "3" ]]; then
        CKPTS+=" 1268"
    elif [[ "${foo:$i:1}" == "4" ]]; then
        CKPTS+=" 1688"
    else
        echo "Unknown checkpoint idx."
    fi
done


if [[ "$5" == "IR" ]]; then
    echo "a"
else
    echo "b" 
fi

echo ${CKPTS}

# if [[ "$1" == "cot" ]]; then
#     TRAINING_DATA_FILE="data/train/processed/cot/cot_data.jsonl"
# elif [[ "$1" == "dolly" ]]; then
#     TRAINING_DATA_FILE="data/train/processed/dolly/dolly_data.jsonl"
# elif [[ "$1" == "flan_v2" ]]; then
#     TRAINING_DATA_FILE="data/train/processed/flan_v2/flan_v2_data.jsonl"
# elif [[ "$1" == "oasst1" ]]; then
#     TRAINING_DATA_FILE="data/train/processed/oasst1/oasst1_data.jsonl"
# else
#     echo "Unknown training data task."
# fi

echo $TRAINING_DATA_NAME $CKPT $DATA_SEED
