#! /bin/bash
export PYTHONPATH="${PYTHONPATH}:$PWD"

python run_visualization.py \
    --lang python \
    --model microsoft/graphcodebert-base \
    --folder graphcodebert \
    --model_type roberta \
    --layer 4 \
    --rank 128 \
    --device 0 &

python run_visualization.py \
    --lang python \
    --model microsoft/codebert-base \
    --folder codebert-baseline \
    --model_type roberta \
    --layer 12 \
    --rank 128 \
    --device 1 &

wait