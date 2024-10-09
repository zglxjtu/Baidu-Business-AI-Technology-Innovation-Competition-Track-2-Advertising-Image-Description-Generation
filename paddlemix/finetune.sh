MODEL_NAME="qwen-vl/qwen-vl-chat-7b"
MASTER='127.0.0.1:8080'
DATA="sft_examples.json"

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node 4 \
examples/qwen_vl/finetune.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA} \
    --dtype 'bfloat16' \
    --fix_vit True \
    --output_dir output_qwen_vl \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --evaluation_strategy "steps" \
    --per_device_eval_batch_size 1 \
    --eval_steps 1000 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --sharding "stage2" \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 4 \
    --pipeline_parallel_degree 1