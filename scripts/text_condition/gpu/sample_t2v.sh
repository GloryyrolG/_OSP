MODEL_PTH="exps/webvid10m_fps8_ft/checkpoint-5000/model_ema"
CUDA_VISIBLE_DEVICES=0 python \
opensora/sample/sample_t2v.py \
    --model_path $MODEL_PTH \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path models/vae \
    --save_img_path "${MODEL_PTH}/test" \
    --fps 8 \
    --guidance_scale 2.5 \
    --num_sampling_steps 20 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"
    # --fps 24 \