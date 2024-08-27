#!/bin/bash

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/data1/home/ycx/workspace/HF/MiniCPM-V-2_6" # 从HuggingFace 或者魔塔下载的官方 MiniCPM-V-2_6路径
DATA="/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/train_set.json" # 训练集路径
EVAL_DATA="/data1/home/ycx/ycxGit/MiniCPM-V-2_6-OD/dataset/test_set.json" # 测试集路径

LLM_TYPE="qwen2" 
# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
#if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE=llama3
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
/home/ycx/anaconda3/envs/MiniCPMV/bin/torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --output_dir output/output__lora \
    --logging_dir output/output_lora \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard" # wandb
