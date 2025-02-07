export PATH=/mnt/data/rongyu/miniconda3/bin:$PATH
source activate /mnt/data/rongyu/miniconda3/envs/osp
cd /mnt/data/rongyu/projects/Open-Sora-Plan

EXP=motionxpp_crop_ft_from93x720p
echo running $EXP
python \
    opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --data "scripts/train_data/merge_data.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path ospv120/vae \
    --sample_rate 1 \
    --num_frames 29 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
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
    --use_image_num 0 \
    --tile_overlap_factor 0.125 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --ema_decay 0.999 \
    --enable_tiling \
    --speed_factor 1.0 \
    --group_frame \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --output_dir="exps/$EXP" \
    --enable_tracker \
    --guidance_scale 7.5 \
    --latent_pose none \
    --crop "true" \
    --pretrained ospv120/93x720p/diffusion_pytorch_model.safetensors \
    # --resume_from_checkpoint="latest" \
