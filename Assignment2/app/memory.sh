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

while kill -0 "$PID" 2> /dev/null; do
    # TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

    # Get total memory usage (MiB)
    TOTAL_MEM=$(cat /proc/meminfo | grep MemTotal | awk '{print $2}')
    # TOTAL_MEM_MIB=$(grep MemTotal "$TOTAL_MEM_INFO" | awk '{print $2 / 1024}')
    FREE_MEM=$(cat /proc/meminfo | grep MemFree | awk '{print $2}')
    # FREE_MEM_MIB=$(grep MemFree "$TOTAL_MEM_INFO" | awk '{print $2 / 1024}')
    # USED_MEM=$(echo "$TOTAL_MEM - $FREE_MEM")

  # Get PID memory usage (KB)
#   if [ -f /proc/$PID/status ]; then
    PID_MEM_KB=$(grep VmRSS /proc/$PID/status | awk '{ print $2 }')
#   else
#     PID_MEM_KB="N/A"
#   fi

    echo "$PROGRAM_ARGS,$TOTAL_MEM,$FREE_MEM,$PID_MEM_KB" >> "$OUTPUT_FILE"
    sleep 1
done