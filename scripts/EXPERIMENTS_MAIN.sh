#!/bin/bash

# include config and some utils
source ./config.sh
source ./utils.sh

export USE_CONDA_ENV=CeDiRNet-py3.8
export DISABLE_X11=0

centernet_filename="${ROOT_DIR}/models/localization_checkpoint.pth"

DO_SYNT_TRAINING=True # step 0: pretraining on syntetic data (MuJoCo) 
DO_REAL_TRAINING=True # step 1: training on real-world data (ViCoS Towel Dataset)
DO_EVALUATION=True    # step 3: evaluate

# assuming 4 GPUs available on localhost
GPU_LIST=("localhost:0" "localhost:1" "localhost:2" "localhost:4")
GPU_COUNT=${#GPU_LIST[@]}


########################################
# PRETRAINING on synthetic data only
########################################

if [[ "$DO_SYNT_TRAINING" == True ]] ; then
  s=0
  for db in "mujoco"; do
    export DATASET=$db
    for backbone in "tu-convnext_base" "tu-convnext_large" ; do
      for epoch in 10; do 
        for depth in off on; do
          if [[ "$depth" == off ]] ; then
            export USE_DEPTH=False
          elif [[ "$depth" == on ]] ; then
            export USE_DEPTH=True
          fi
          SERVERS=${GPU_LIST[$((s % GPU_COUNT))]} ./run_distributed.sh python train.py --config \
                                      model.kwargs.backbone=$backbone \
                                      n_epochs=$epoch \
                                      train_dataset.batch_size=4 \
                                      train_dataset.workers=16 \
                                      "pretrained_center_model_path=$centernet_filename" \
                                      display=False save_interval=1 skip_if_exists=True &
          s=$((s+1))
          wait_or_interrupt $GPU_COUNT $s
        done
      done
    done
  done
fi
wait_or_interrupt

########################################
# Training on real data
########################################

if [[ "$DO_REAL_TRAINING" == True ]] ; then
  s=0

  for db in "vicos_towel"; do
    export DATASET=$db
    export TRAIN_SIZE=768
    for backbone in "tu-convnext_base" "tu-convnext_large" ; do
      for epoch in 10; do
        for depth in off on; do 
          if [[ "$depth" == off ]] ; then
            export USE_DEPTH=False
            depth_str=False
          elif [[ "$depth" == on ]] ; then
            export USE_DEPTH=True 
            depth_str=True
          fi                    
          PRETRAINED_CHECKPOINT="${OUTPUT_DIR}/mujoco/backbone=${backbone}/num_train_epoch=10/depth=$depth_str/multitask_weight=uw/checkpoint.pth"
          SERVERS=${GPU_LIST[$((s % GPU_COUNT))]} ./run_distributed.sh python train.py --config \
                                    model.kwargs.backbone=$backbone \
                                    n_epochs=$epoch \
                                    "pretrained_center_model_path=$centernet_filename" \
                                    "pretrained_model_path=$PRETRAINED_CHECKPOINT" \
                                    display=True skip_if_exists=True "save_interval=1" &
          s=$((s+1))
          wait_or_interrupt $GPU_COUNT $s
        done
      done
    done
  done
fi
wait_or_interrupt

########################################
# Evaluating on test data
########################################

if [[ "$DO_EVALUATION" == True ]] ; then

  # FOR FULL EVAL
  DISPLAY_ARGS="display=False display_to_file_only=True skip_if_exists=True"

  s=0
  export DISABLE_X11=0
  for db in "vicos_towel"; do
    for cfg_subname in ""; do # for exclusively unseen objects set to "novel_object=bg+cloth" (or to "novel_object=cloth" "novel_object=bg")
      export DATASET=$db
      export TRAIN_SIZE=768
      export TEST_SIZE=768
      for backbone in "tu-convnext_base" "tu-convnext_large" ; do
        for epoch_train in 10; do
          ALL_EPOCH=("") # set to ALL_EPOCH=("" _002 _004 _006 _008) to evaluate every second epoch
          for epoch_eval in "${ALL_EPOCH[@]}"; do
            for depth in off on; do
              if [[ "$depth" == off ]] ; then
                export USE_DEPTH=False
              elif [[ "$depth" == on ]] ; then
                export USE_DEPTH=True
              fi
              # run center model pre-trained on weakly-supervised
              SERVERS=${GPU_LIST[$((s % GPU_COUNT))]} ./run_distributed.sh python test.py  --cfg_subname="$cfg_subname" --config \
                                        eval_epoch=$epoch_eval \
                                        model.kwargs.backbone=$backbone \
                                        train_settings.n_epochs=$epoch_train \
                                        "center_checkpoint_path=$centernet_filename" \
                                        center_checkpoint_name_list=None \
                                        $DISPLAY_ARGS &

              s=$((s+1))
              wait_or_interrupt $GPU_COUNT $s
            done
          done
        done
      done
    done
  done
fi
wait