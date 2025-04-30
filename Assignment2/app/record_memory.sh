#!/bin/bash

OUTPUT_FILE="$1"

if [ -z "$OUTPUT_FILE" ]; then
  echo "Usage: $0 <OUTPUT_FILE> [PROGRAM_ARGS...]"
  exit 1
fi

shift 1
PROGRAM_ARGS=$(printf "%s," "$@")
PROGRAM_ARGS=${PROGRAM_ARGS%,}

top -b -n 1 | awk -v args="$PROGRAM_ARGS" '/MiB Mem/ { print args "," $0 }' >> "$OUTPUT_FILE"
