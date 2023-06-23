#! /bin/bash
export PYTHONPATH="${PYTHONPATH}:$PWD"

NUM_DEVICES=2
DEVICE=0
while ((DEVICE < NUM_DEVICES))
do
    python run_experiments_rq123_replicate.py \
        --langs python \
        --models microsoft/codebert-base microsoft/codebert-base microsoft/graphcodebert-base microsoft/unixcoder-base microsoft/unixcoder-base-unimodal microsoft/unixcoder-base-nine microsoft/unixcoder-base microsoft/unixcoder-base-unimodal microsoft/unixcoder-base-nine \
        --folders codebert-baseline codebert0 graphcodebert unixcoder unixcoder-unimodal unixcoder-nine unixcoder-pretrain unixcoder-unimodal-pretrain unixcoder-nine-pretrain \
        --model_types roberta roberta roberta unilm unilm unilm unilm_pretrain unilm_pretrain unilm_pretrain \
        --num_devices $NUM_DEVICES \
        --device $DEVICE &

    DEVICE=$((DEVICE+1))
done

wait