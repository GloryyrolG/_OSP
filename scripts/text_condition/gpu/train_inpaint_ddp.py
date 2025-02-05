import os
import torch
NNODES = os.environ.get('WORLD_SIZE', 1)
NODE_RANK = os.environ.get('RANK', 0)
MASTER_ADDR = os.environ.get('MASTER_ADDR', '127.0.0.1')
MASTER_PORT = os.environ.get('MASTER_PORT', '12345')
LOCAL_GPU_NUM = torch.cuda.device_count()
GLOBAL_GPU_NUM = LOCAL_GPU_NUM * int(NNODES)
batch_size = 1
print(NNODES, NODE_RANK, GLOBAL_GPU_NUM, LOCAL_GPU_NUM, MASTER_ADDR, MASTER_PORT)
#DeepFloyd/t5-v1_1-xxl
command = 'export ENTITY="linbin" &&\
    export HF_DATASETS_OFFLINE=1 &&\
    export TRANSFORMERS_OFFLINE=1 &&\
    export PDSH_RCMD_TYPE=ssh &&\
    export GLOO_SOCKET_IFNAME=bond0 &&\
    export NCCL_SOCKET_IFNAME=bond0 &&\
    export NCCL_SOCKET_IFNAME=en,eth,em,bond &&\
    export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1 &&\
    export NCCL_IB_GID_INDEX=3 &&\
    export NCCL_IB_TC=162 &&\
    export NCCL_IB_TIMEOUT=22 &&\
    export NCCL_PXN_DISABLE=0 &&\
    export NCCL_IB_QPS_PER_CONNECTION=4 &&\
    export NCCL_ALGO=Ring &&\
    export OMP_NUM_THREADS=1 &&\
    export MKL_NUM_THREADS=1 &&\
    accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config_ddp.json --main_process_ip={} --main_process_port={} --num_machines={} --num_processes={} --machine_rank={}\
    opensora/train/train_inpaint.py \
    --model OpenSoraInpaint-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "../cache_dir" \
    --dataset inpaint \
    --data "scripts/train_data/merge_data.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "ospv120/vae" \
    --sample_rate 1 \
    --num_frames 29 \
    --use_image_num 0 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size={} \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=1000 \
    --allow_tf32 \
    --model_max_length 512 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --i2v_ratio 0.4 \
    --transition_ratio 0.4 \
    --v2v_ratio 0.1 \
    --clear_video_ratio 0.0 \
    --default_text_ratio 0.5 \
    --use_rope \
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --group_frame \
    --enable_tracker \
    --num_sampling_steps 20 \
    --output_dir exps/anim_nop_mtsk_idol_mfuse_stg1 \
    --latent_pose none \
    --crop "true" \
    --multitask idol \
    --skips none \
    --mullev none \
    --train_stage rg \
    --guidance_scale 7.5 \
    --pretrained_transformer_model_path "exps/motionxpp_nop_multask_sk_sep/checkpoint-30000/model_ema" \
    '.format(MASTER_ADDR, MASTER_PORT, NNODES, GLOBAL_GPU_NUM, NODE_RANK, batch_size)
# --pretrained_transformer_model_path "ospv120/29x480p"
# "ospv120/93x480p_i2v"
# --resume_from_checkpoint="latest"
print(command)
os.system(command)