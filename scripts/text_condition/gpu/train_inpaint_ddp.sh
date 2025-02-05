# sudo apt update
# sudo apt install tmux
# tmux new -s train
# tmux set mouse on
export PATH=/mnt/data/rongyu/miniconda3/bin:$PATH
source activate /mnt/data/rongyu/miniconda3/envs/osp
cd /mnt/data/rongyu/projects/Open-Sora-Plan
python scripts/text_condition/gpu/train_inpaint_ddp.py