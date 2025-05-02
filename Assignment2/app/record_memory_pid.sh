#!/bin/bash

PID="$1"
OUTPUT_FILE="$2"

# if the file is sequential this are the columns names
# input;VIRT;RES;SHR;
# else 
# input;thread;VIRT;RES;SHR;

if [ -z "$PID" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Usage: $0 <PID> <OUTPUT_FILE> [PROGRAM_ARGS...]"
  exit 1
fi

shift 2  
PROGRAM_ARGS=$(printf "%s," "$@")
PROGRAM_ARGS=${PROGRAM_ARGS%,}

while kill -0 "$PID" 2> /dev/null; do
  # top -b -n 1 -p "$PID" | awk -v args="$PROGRAM_ARGS" \
  #   '/^[[:space:]]*'"$PID"'/ {print args "," $5 "," $6 "," $7}' >> "$OUTPUT_FILE"
  # sleep 3
    if [ -f /proc/$PID/status ]; then
        MEM_KB=$(grep VmRSS /proc/$PID/status | awk '{ print $2 }')
        echo "$PROGRAM_ARGS, $MEM_KB" >> "$OUTPUT_FILE"
    fi
    sleep 1
done