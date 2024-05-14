#!/bin/bash

for fasta_file in /home/mark/other_scripts/split_fasta_files_spliceator/*.fa; do
  # Create the name of the output file by replacing the extension
  output_file="/home/mark/other_scripts/split_fasta_files_spliceator/$(basename "$fasta_file" .fa)_Spliceator_results.csv"

  # Check if the output file already exists
  if [ -f "$output_file" ]; then
    echo "Output for $fasta_file already exists, skipping..."
    continue # Skip the rest of the loop and go to the next fasta file
  fi

  # Run Spliceator if the output file does not exist
  ./Spliceator -f "$fasta_file" -o "$output_file"
done
