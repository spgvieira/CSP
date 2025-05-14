#!/bin/bash

PID="$1"
REPEAT_NUMBER="$2"
OUTPUT_FILE="$3"

shift 3
PROGRAM_ARGS=$(printf "%s," "$@")
PROGRAM_ARGS=${PROGRAM_ARGS%,}

# timestamp,program_args,run_number,total_mem_mib,pid_mem_kb
# program_args are the algorithm argument and the number of threas (if relevant)
while kill -0 "$PID" 2> /dev/null; do
    TIMESTAMP=$(date +"%m-%d %H:%M:%S")
    FREE_MEM=$(cat /proc/meminfo | grep MemFree | awk '{print $2}')
    PID_MEM_KB=$(grep VmRSS /proc/$PID/status | awk '{ print $2 }')
    echo "$TIMESTAMP,$REPEAT_NUMBER,$PROGRAM_ARGS,$FREE_MEM,$PID_MEM_KB" >> "$OUTPUT_FILE"
    sleep 1
done 
