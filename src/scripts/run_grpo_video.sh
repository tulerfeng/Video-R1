cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12351" \
    src/open_r1/grpo.py \
    --output_dir "YOUR_PATH/log_dvd" \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "YOUR_PATH/data/train_dvd.jsonl" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-7B-Video-dvd \
    --save_steps 100 \
    --max_grad_norm 20 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
