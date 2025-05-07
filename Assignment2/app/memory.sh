#!/bin/bash

PID="$1"
OUTPUT_FILE="$2"

if [ -z "$PID" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <PID> <OUTPUT_FILE> [PROGRAM_ARGS...]"
exit 1
fi

shift 2
PROGRAM_ARGS=$(printf "%s," "$@")
PROGRAM_ARGS=${PROGRAM_ARGS%,}

# echo "timestamp,program_args,total_mem_mib,pid_mem_kb" >> "$OUTPUT_FILE"
if [[ "$PID" != "baseline" ]]; then
    while kill -0 "$PID" 2> /dev/null; do
        FREE_MEM=$(cat /proc/meminfo | grep MemFree | awk '{print $2}')
        PID_MEM_KB=$(grep VmRSS /proc/$PID/status | awk '{ print $2 }')
        echo "$PROGRAM_ARGS,$FREE_MEM,$PID_MEM_KB" >> "$OUTPUT_FILE"
        sleep 1
    done
else 
    END_TIME=$(date -d "now + 3 minutes" +%s)

    while [ $(date +%s) -lt "$END_TIME" ]; do
        FREE_MEM=$(cat /proc/meminfo | grep MemFree | awk '{print $2}')
        echo "$FREE_MEM" >> "$OUTPUT_FILE"
        sleep 1
    done
fi
