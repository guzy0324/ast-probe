#! /bin/bash
export PYTHONPATH="${PYTHONPATH}:$PWD"

NUM_DEVICES=2
DEVICE=0
while ((DEVICE < NUM_DEVICES))
do
    python run_experiments_rq4_replicate.py \
        --langs python \
        --models microsoft/graphcodebert-base microsoft/unixcoder-base microsoft/unixcoder-base-unimodal microsoft/unixcoder-base-nine \
        --folders graphcodebert unixcoder unixcoder-unimodal unixcoder-nine \
        --model_types roberta unilm unilm unilm \
        --num_devices $NUM_DEVICES \
        --device $DEVICE &

    DEVICE=$((DEVICE+1))
done

wait