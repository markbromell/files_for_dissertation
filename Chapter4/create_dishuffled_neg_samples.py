import subprocess
import os

# Define the input FASTA file
input_fasta_file = "/path/to/exon_sequences.fa"

# Define the directory to save the shuffled FASTA files
output_dir = "dishuffle_shuffled_fasta_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_and_shuffle_fasta(input_fasta, output_directory):
    with open(input_fasta, 'r') as fasta_file:
        content = fasta_file.read().strip()
        # Ensure splitting does not lead to empty entries
        entries = [entry for entry in content.split('>') if entry.strip()]
        for entry in entries:
            header, *sequence = entry.strip().split('\n')
            sequence = '\n'.join(sequence).strip()
            if not sequence:  # Skip entries without a sequence
                print(f"Skipping empty sequence for header: {header}")
                continue
            output_filename = f"{header.split()[0]}.fa"
            output_path = os.path.join(output_directory, output_filename)

            with open(output_path, 'w') as out_file:
                out_file.write(f">{header}\n{sequence}\n//\n")

            print(f"Writing to {output_path}")

            # Run the dishuffle command
            shuffled_output_path = output_path.replace('.fa', '.dishuf')
            try:
                subprocess.run(f"perl dishuffle.pl {output_path} > {shuffled_output_path}", shell=True, check=True)
                print(f"Shuffled file created: {shuffled_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error shuffling file {output_path}: {e}")

# Call the function with your input FASTA file and output directory
parse_and_shuffle_fasta(input_fasta_file, output_dir)

print(f"Shuffled FASTA files are saved in {output_dir}")
