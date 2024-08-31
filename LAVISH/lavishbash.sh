#!/bin/bash

#SBATCH --job-name=trial0
#SBATCH --account=ec29
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=ifi_accel
# #SBATCH --gres=gpu:1
##SBATCH --qos=devel
# #SBATCH --mem=32G      
#SBATCH --gpus=rtx30:1

module purge
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
source ./envs/lavish/bin/activate

DIR=./LAVISH/AVQA2

# python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=./LAVISH/AVQA2/data/feats/vggish --batch-size=1 --early_stop=5 --epochs=30 --is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --lr=8e-05 --lr_block=3e-06 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=/fp/projects01/ec12/annammc/tismir/LAVISH/AVQA2/data/feats/res18_14x14 --video_dir=/fp/projects01/ec12/annammc/tismir/LAVISH/AVQA2/data/frames --wandb=1 --mode train
python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir ./LAVISH/AVQA2/data/feats/vggish --batch-size=1 --early_stop=5 --epochs=30 --is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir /fp/projects01/ec12/annammc/tismir/LAVISH/AVQA2/data/feats/res18_14x14 --wandb=1 --mode test
# python ${DIR}/grounding_gen/main_grd_gen.py --batch-size 16 --epochs 30 --lr 8e-05 --mode train --label_train ${DIR}/data/json/avqa-train.json --label_val ${DIR}/data/json/avqa-val.json --label_test ${DIR}/data/json/avqa-test.json --model_save_dir ${DIR}/grounding_gen/models_grounding_gen/
