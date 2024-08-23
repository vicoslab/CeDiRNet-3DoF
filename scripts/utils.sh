#!/bin/bash

function cleanup() {
  echo "Terminated. Cleaning up .."
  child_ids=$(pgrep -P $$ | xargs echo | tr " " ,)
  # kill all child processes
  pkill -P $child_ids
  pkill $$
  exit 0
}

function wait_or_interrupt() {
  # set to kill any child processes if parent is interupted
  #trap "pkill -P $child_ids && pkill $$ && echo exit && exit 0" SIGINT
  trap cleanup SIGINT
  # now wait
  if [ -z "$1" ] ; then
    wait
  elif [ -n "$1" ] && [ -n "$2" ] ; then
    MAX_CAPACITY=$1
    INDEX=$2
    # wait only if INDEX mod MAX_CAPACITY == 0
    if [ $((INDEX % MAX_CAPACITY)) -eq 0 ] ; then
      wait
    fi
  else
    # wait if more child processes exist than allowed ($1 is the number of allowed children)
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
  fi
}