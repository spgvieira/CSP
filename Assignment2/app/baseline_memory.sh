#!/bin/bash

OUTPUT_FILE="$1"

END_TIME=$(date -d "now + 3 minutes" +%s)

while [ $(date +%s) -lt "$END_TIME" ]; do
    FREE_MEM=$(cat /proc/meminfo | grep MemFree | awk '{print $2}')
    echo "$FREE_MEM" >> "$OUTPUT_FILE"
    sleep 1
done