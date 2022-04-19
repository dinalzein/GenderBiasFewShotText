#!/usr/bin/env bash
cd $HOME/Projects/GenderBiasFewShotText
# Activate environment
source .venv/bin/activate
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
command -v nvidia-smi >/dev/null && {
    echo "GPU Devices:"
    nvidia-smi
} || {
    :
}

PYTHONPATH=. python models/matching/matchingnet.py $@
