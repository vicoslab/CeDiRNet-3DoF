#!/bin/bash

CMD_ARGS=$@

job_main() {
  # initial call that will delegate job into servers
  # SERVERS env var should be list of servers ang gpu ids per each server (e.g. "donbot:0,1,2,3 morbo:2,3 calculon:1,0"
  # first count number of servers and GPUs to get the world size
  num_gpus="${SERVERS//[^,]}"
  num_gpus="${#num_gpus}"

  num_servers="${SERVERS//[^ ]}"
  num_servers="${#num_servers}"

  WORLD_SIZE=$((num_gpus+num_servers+1))
  RANK_OFFSET=0
  MASTER_PORT=$((RANDOM+24000))

  IFS=' ' read -ra ADDR_LIST <<< "$SERVERS"
  for ADDR in "${ADDR_LIST[@]}"; do
    # address is in format: <SEVER_NAME>:<CUDA_VISIBLE_DEVICES> (e.g. donbot:0,1,2,3)
    IFS=':' read -ra NAME_ID <<< "$ADDR"
    SERVER_NAME=${NAME_ID[0]}
    CUDA_VISIBLE_DEVICES=${NAME_ID[1]}

    # set master to first server
    if [ -z "$MASTER_ADDR" ]; then
      MASTER_ADDR=$SERVER_NAME
    fi

    # pass to ssh all needed env vars
    ENVS=""
    ENVS="$ENVS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    ENVS="$ENVS DATASET=$DATASET"
    ENVS="$ENVS USE_DEPTH=$USE_DEPTH"
    if [ -n "$TRAIN_SIZE" ]; then
      ENVS="$ENVS TRAIN_SIZE=$TRAIN_SIZE"
    fi
    if [ -n "$TRAIN_SIZE_WIDTH" ]; then
      ENVS="$ENVS TRAIN_SIZE_WIDTH=$TRAIN_SIZE_WIDTH"
    fi
    if [ -n "$TRAIN_SIZE_HEIGHT" ]; then
      ENVS="$ENVS TRAIN_SIZE_HEIGHT=$TRAIN_SIZE_HEIGHT"
    fi
    if [ -n "$TEST_SIZE" ]; then
      ENVS="$ENVS TEST_SIZE=$TEST_SIZE"
    fi
    if [ -n "$TEST_SIZE_WIDTH" ]; then
      ENVS="$ENVS TEST_SIZE_WIDTH=$TEST_SIZE_WIDTH"
    fi
    if [ -n "$TEST_SIZE_HEIGHT" ]; then
      ENVS="$ENVS TEST_SIZE_HEIGHT=$TEST_SIZE_HEIGHT"
    fi
    ENVS="$ENVS ENABLE_6DOF=$ENABLE_6DOF"
    ENVS="$ENVS ENABLE_EULER=$ENABLE_EULER"
    
    ENVS="$ENVS MASTER_PORT=$MASTER_PORT"
    ENVS="$ENVS MASTER_ADDR=$MASTER_ADDR"
    ENVS="$ENVS WORLD_SIZE=$WORLD_SIZE"
    ENVS="$ENVS RANK_OFFSET=$RANK_OFFSET"
    if [ -n "$USE_CONDA_ENV" ]; then
      ENVS="$ENVS USE_CONDA_ENV=$USE_CONDA_ENV"
    fi

    export ENVS
    export SERVER_NAME
    # run ssh connection in child background process
    RUN_TASK=2 $(realpath $0) $CMD_ARGS &

    # increase world rank offset by the number of gpus
    num_gpus="${CUDA_VISIBLE_DEVICES//[^,]}"
    num_gpus="${#num_gpus}"
    RANK_OFFSET=$((RANK_OFFSET + num_gpus + 1))
  done

  wait
}

ssh_main() {
  SSH_ARGS="-t -t -o StrictHostKeyChecking=no"
  if [ "${DISABLE_X11}" != "1" ]; then
    SSH_ARGS="-Y $SSH_ARGS"
  fi
  # call main task function on server (use -t -t to allow exiting remote process in interuption)
  exec ssh $SSH_ARGS $SERVER_NAME RUN_TASK=1 $ENVS $(realpath $0) $(printf "%q " "$CMD_ARGS")
}

task_main() {
  # Set up signal trap to catch Ctrl+C
  trap "exit" SIGINT

  # set up env vars
  source "$(dirname $0)/config.sh"

  echo "NODE=$HOSTNAME"
  echo "WORLD_SIZE=$WORLD_SIZE"
  echo "RANK_OFFSET=$RANK_OFFSET"

  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  echo "master=$MASTER_ADDR"
  ###################################################

  cd $SOURCE_DIR
  $CMD_ARGS
}

if [ "${RUN_TASK}" = "2" ]; then
  ssh_main
elif [ "${RUN_TASK}" = "1" ]; then
  task_main
else
  job_main
fi