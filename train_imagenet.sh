#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Remember to configure the conda environment to pytorch 0.4.1
case $HOST in
"Hydrogen")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD="/home/elliot/.conda/envs/pytorch_041/bin/tensorboard"
    data_path='/opt/imagenet' #dataset path
    ;;
"alpha")
    PYTHON="/home/elliot/anaconda3/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/anaconda3/envs/pytorch_041/bin/tensorboard'
    ;;
"Helium")
    PYTHON="/home/elliot/.conda/envs/pytorch_041/bin/python" # python environment
    TENSORBOARD='/home/elliot/.conda/envs/pytorch_041/bin/tensorboard'
    data_path='/opt/imagenet/imagenet_compressed' #dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display= false # enable tensorboard display
model=quan_resnet18b_fq_lq
dataset=imagenet
epochs=50
batch_size=256
optimizer=Adam
quantize=resume

tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

# pretrained_model=/home/elliot/Documents/aaai_2019/save/imagenet_quan_resnet18b_ff_lf_50_Adam_tern/checkpoint.pth.tar


$PYTHON main_dc.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize} \
    --epochs ${epochs} --learning_rate 0.0001 \
    --optimizer ${optimizer} \
	--schedule 25 30 35  --gammas 0.2 0.2 0.5 \
    --batch_size ${batch_size} --workers 16 --ngpu 4 \
    --print_freq 50 --decay 0.000005 --momentum 0.9 \
    # --resume ${pretrained_model} \
    # --evaluate
