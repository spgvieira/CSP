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

MEM_PEAK_KB=$(grep VmPeak /proc/$PID/status | awk '{ print $2 }')
echo "$PROGRAM_ARGS, $PID, $MEM_PEAK_KB" >> "$OUTPUT_FILE"
