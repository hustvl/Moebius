#! /bin/bash
WORK_DIR="${WORK_DIR:-.}"


############## set runtime args ###############
echo "- set runtime args ----------------"
NUM_MACHINES=${NUM_MACHINES:-1}
NUM_GPUS_PER_MACHINE=${NUM_GPUS_PER_MACHINE:-1}
NUM_PROCESSES=$((NUM_MACHINES * NUM_GPUS_PER_MACHINE))
# MAIN_PROC_IP=${MAIN_PROC_IP:-""}
# MAIN_PROC_PORT=${MAIN_PROC_PORT:-8008}

# main_ID=${main_ID:-0}
# end=$((NUM_MACHINES - 1))


# echo "doing clean proc..."
# pkill python
# pkill accelerate


# check runtime env
# echo "main node ID: $main_ID"
# echo "main node IP: $MAIN_PROC_IP"
echo "num machines: $NUM_MACHINES"
echo "num gpus/machine: $NUM_GPUS_PER_MACHINE"
echo "num gpu procs: $NUM_PROCESSES"
sleep 5s

############## set experiment args ###############
echo "- set experiment args ----------------"
CONFIG_FILE=$1
EXP_NAME=$(echo "$CONFIG_FILE" | sed -E 's|^(./)?config/||; s|\.sh$||')
# check exp name
echo "exp name: $EXP_NAME" && export EXP_NAME=$EXP_NAME
echo "exp config: $CONFIG_FILE" && source $CONFIG_FILE # get TRAIN_ARGS,OUTPUT_DIR,OUTPUT_DIR_EXP_NAME from $CONFIG_FILE 
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR_EXP_NAME
mkdir -p "$OUTPUT_DIR_EXP_NAME/log"
sleep 5s

# check exp config
PY_TRAINER=${PY_TRAINER:-"train_ddp.py"}
echo "training python script: $PY_TRAINER"
printf "TRAIN_ARGS =\n"
for arg in $TRAIN_ARGS; do
    printf "  %s\n" "$arg"
done
sleep 5s

############## set cmdline and launch ###############
echo "- set cmdline and launch ----------------"
ACC_CMD="accelerate launch \
        --num_cpu_threads_per_process 16 \
        --num_processes ${NUM_PROCESSES} \
        --num_machines ${NUM_MACHINES}"
TRAIN_CMD="${PY_TRAINER} ${TRAIN_ARGS}"
# TEE_CMD=" 2>&1 | tee ${OUTPUT_DIR_EXP_NAME}/log/$(basename "$EXP_NAME")_`date "+%Y.%m.%d-%H:%M:%S"`.log"



$ACC_CMD --machine_rank 0 $TRAIN_CMD  2>&1 | tee ${OUTPUT_DIR_EXP_NAME}/log/$(basename "$EXP_NAME")_`date "+%Y.%m.%d-%H:%M:%S"`.log
