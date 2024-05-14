import sys
import numpy as np
import os
import gc
from tensorflow.keras.models import load_model

# Set seed for reproducibility
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Load the acceptor model only
acceptor_model = load_model("/path/to/donor/or/acceptor/model.h5")

# Nucleotide dictionary for one-hot encoding
nucleotide_dict = {
    "A": [1, 0, 0, 0],
    "G": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "a": [1, 0, 0, 0],
    "g": [0, 1, 0, 0],
    "c": [0, 0, 1, 0],
    "t": [0, 0, 0, 1],
    "N": [0.25, 0.25, 0.25, 0.25],
    "n": [0.25, 0.25, 0.25, 0.25]
}

def read_fasta_sequences_in_chunks(fasta_file_path, start, end):
    count = 0
    sequences = []
    with open(fasta_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequences:
                    count += 1
                    if count > end:
                        break
                    if count >= start:
                        yield sequences
                    sequences = []
            else:
                sequences.append(line)
        if sequences and count >= start and count <= end:
            yield sequences

def one_hot_encode_list(seq_list, nucleotide_dict):
    max_len = max(len(seq) for seq in seq_list)
    seq_array = np.zeros((len(seq_list), max_len, 4), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, nt in enumerate(seq):
            if nt in nucleotide_dict:
                seq_array[i, j, :] = nucleotide_dict[nt]
    return seq_array

def process_and_save_predictions(model, fasta_file_path, output_file_path, nucleotide_dict, start, end):
    with open(output_file_path, 'a') as output_file:  # Open in append mode
        for seq_chunk in read_fasta_sequences_in_chunks(fasta_file_path, start, end):
            encoded_chunk = one_hot_encode_list(seq_chunk, nucleotide_dict)
            predictions = model.predict(encoded_chunk)
            for prediction in predictions:
                output_file.write(f"{prediction}\n")
            del encoded_chunk, predictions
            gc.collect()

def main(fasta_file_path, output_file_path, start, end):
    process_and_save_predictions(acceptor_model, fasta_file_path, output_file_path, nucleotide_dict, int(start), int(end))

if __name__ == '__main__':
    main(*sys.argv[1:])
