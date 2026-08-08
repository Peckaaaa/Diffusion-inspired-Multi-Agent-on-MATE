#!/bin/bash
which python

env="mamujoco"
map_name="HalfCheetah-v2"
agent_conf="6x1"
steps=1000000
seed=1
cuda_device=0

# --ce_for_av
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --n_workers 1 --env ${env} --env_name ${map_name} --agent_conf ${agent_conf} --policy_class gaussian --seed ${seed} --steps $steps --mode disabled \
                                                    --temperature 1.0 --sample_temp 20 --state_decoder_type 1
