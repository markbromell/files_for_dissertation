#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="script_to_run_SpliceFinder.py"

# Total number of sequences (assuming each sequence is followed by its ID)
TOTAL_SEQUENCES=$(wc -l < /home/mark_bromell/SpliceFinder/sequences_mb/shuffled_sequences_SpliceFinder_encoded_1of2.txt)
TOTAL_SEQUENCES=$((TOTAL_SEQUENCES / 2)) # Halve the count, as every second line is a sequence

# Number of sequences per batch
BATCH_SIZE=100000

# Loop through the dataset in chunks
for ((i=0; i<TOTAL_SEQUENCES; i+=BATCH_SIZE)); do
    start_idx=$((i * 2)) # Double the index as every second line is processed
    end_idx=$(((i + BATCH_SIZE) * 2))
    if [ $end_idx -gt $((TOTAL_SEQUENCES * 2)) ]; then
        end_idx=$((TOTAL_SEQUENCES * 2))
    fi

    echo "Processing sequences from $((start_idx / 2)) to $((end_idx / 2))"
    python3 $PYTHON_SCRIPT --start_idx $start_idx --end_idx $end_idx
done
