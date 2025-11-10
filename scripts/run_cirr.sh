#!/bin/bash -l

CUDA_ID=0
seeds=(1)
METHODS=(reward)
DATASETS=(cirr)

MIXTURE_FACTOR_IMG=0.2
MIXTURE_FACTOR_TEXT=0.8
TEACHER_TEXT_FACTOR=0.8
DELTA=0.1
EPSILON=1.0
FUSION_TYPE=lerp
OPTIMIZER=AdamW
LR=0.0005
OPT_STEPS=1
CLIP_SCORE_WEIGHT=2.5
RWD_SAMPLE_K=8
BASELINE_SCORE=RLOO
SELF_REWARD_WEIGHT=1.0

for METHOD in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Running with METHOD=${METHOD}, dataset=${dataset}, seed=${seed}"

        CUDA_VISIBLE_DEVICES=${CUDA_ID} python test_time.py \
        --cfg cfgs/coir/${METHOD}.yaml \
        CORRUPTION.DATASET ${dataset} \
        OPTIM.MIXTURE_FACTOR_TEXT ${MIXTURE_FACTOR_TEXT} \
        REWARD.TEACHER_TEXT_FACTOR ${TEACHER_TEXT_FACTOR} \
        OPTIM.DELTA ${DELTA} \
        OPTIM.EPSILON ${EPSILON} \
        OPTIM.FUSION_TYPE ${FUSION_TYPE} \
        OPTIM.METHOD ${OPTIMIZER} \
        OPTIM.LR ${LR} \
        OPTIM.STEPS ${OPT_STEPS} \
        REWARD.CLIP_SCORE_WEIGHT ${CLIP_SCORE_WEIGHT} \
        REWARD.SAMPLE_K ${RWD_SAMPLE_K} \
        REWARD.SAMPLING_TYPE "temperature" \
        REWARD.TEMPERATURE ${RWD_SAMPLE_K_TEMPERATURE} \
        RNG_SEED ${seed} \
        REWARD.BASEINE_SCORE ${BASELINE_SCORE} \
        MODEL.USE_CLIP True \
        MODEL.ARCH ViT-B-16 \
        REWARD.REWARD_ARCH ViT-L-14 \
        REWARD.USE_SELF_REWARD True \
        REWARD.SELF_REWARD_WEIGHT ${SELF_REWARD_WEIGHT} \
        SAVE_DIR "./output/log" \
        DESC "exp for [${METHOD}] on [${dataset}] with seed [${seed}]"
    done
done