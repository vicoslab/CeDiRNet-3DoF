#!/bin/bash

###################################################
######## LOAD USER-SPECIFIC CONFIG
###################################################

USER_CONFIG_FILE="$(dirname $0)/config_user.sh"
# create config file if it does not exist
if [ ! -f "$USER_CONFIG_FILE" ]; then
  cp "$(dirname $0)/config_user.sh.example" "$USER_CONFIG_FILE"
fi

# include user-specific settings
# shellcheck source=./config_user.sh
source "$USER_CONFIG_FILE"

###################################################
######## ACTIVATE CONDA ENV
###################################################
echo "Loading conda env ..."

USE_CONDA_HOME=${USE_CONDA_HOME:-~/conda}
USE_CONDA_ENV=${USE_CONDA_ENV:-CeDiRNet-dev}

. $USE_CONDA_HOME/etc/profile.d/conda.sh

conda activate $USE_CONDA_ENV
echo "... done - using $USE_CONDA_ENV"

###################################################
######## INPUT/OUTPUT PATH
###################################################

export SOURCE_DIR=${SOURCE_DIR:-$(realpath "$(dirname $0)/../src")}
export OUTPUT_DIR=${OUTPUT_DIR:-$(realpath "$(dirname $0)/../exp")}

###################################################
######## DATASET PATHS
###################################################

export MUJOCO_DIR=${MUJOCO_DIR:-(realpath "$(dirname $0)/../datasets/MuJoCo/")}
export VICOS_TOWEL_DATASET_DIR=${VICOS_TOWEL_DATASET_DIR:-(realpath "$(dirname $0)/../datasets/ViCoSTowelDataset/")}

###################################################
######## DATA PARALLEL SETTINGS
###################################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=1
#export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1}
