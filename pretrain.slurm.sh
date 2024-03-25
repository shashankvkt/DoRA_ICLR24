#!/bin/bash

#SBATCH --job-name=dora
#SBATCH --time=100:00:00
#SBATCH --account=ofq@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4
##SBATCH  -C v100-32g
##SBATCH  -C a100
#SBATCH --ntasks=8                     # total number of GPUs
#SBATCH --ntasks-per-node=8            # GPUs per node
#SBATCH --nodes=1                      # reserving n node
#SBATCH --gres=gpu:8                   # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=3              # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --hint=nomultithread           # hyperthreading is deactivated

##SBATCH --output=/gpfswork/rech/ofq/uco38ei/WT_code/DoRA/logfiles/single_vid/pretrain/amsterdam_8frame.out
##SBATCH --error=/gpfswork/rech/ofq/uco38ei/WT_code/DoRA/logfiles/single_vid/pretrain/amsterdam_8frame.err


module purge
module load python/3.8.2
conda activate mot_ibot


set -x


echo "Training Start"

python -m torch.distributed.launch --nproc_per_node=8 main.py --arch vit_small --data_path /gpfsstore/rech/ofq/uco38ei/Datasets/WT_videos/amsterdam/amsterdam.mp4 \
  --output_dir /gpfsstore/rech/ofq/uco38ei/weights_WTours/DoRA/amsterdam_8frames/ --optimizer adamw \
  --use_bn_in_head False --out_dim 65536 --batch_size_per_gpu 6 --local_crops_number 6 --epochs 100  \
  --num_workers 10 --lr 0.0005 --min_lr 0.00001  --norm_last_layer False  \
  --warmup_teacher_temp_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4 \
  --frame_per_clip 8 --step_between_clips 60

echo "Training Done"
