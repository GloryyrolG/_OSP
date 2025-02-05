export PATH=/mnt/data/rongyu/miniconda3/bin:$PATH
source activate /mnt/data/rongyu/miniconda3/envs/osp
cd /mnt/data/rongyu/projects/Open-Sora-Plan

EXP=motionxpp_i2v_from93x480pi2v
echo running $EXP
# accelerate launch \
#     --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
# CUDA_VISIBLE_DEVICES=1 taskset -c 12-23
python \
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
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
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
    --max_train_steps=100000 \
    --checkpointing_steps=1000 \
    --num_sampling_steps 20 \
    --output_dir exps/$EXP \
    --latent_pose none \
    --crop "false" \
    --guidance_scale 7.5 \
    --resume_from_checkpoint="latest" \
    # --pretrained_transformer_model_path "ospv120/93x480p_i2v" \
