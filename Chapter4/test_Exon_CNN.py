import numpy as np
from Bio import SeqIO
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

np.random.seed(42)
tf.random.set_seed(42)

def build_kmer_index(k):
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    kmer_index = {kmer: i for i, kmer in enumerate(kmers)}
    return kmer_index

def sequence_to_kmer_ids(sequence, k, kmer_index):
    return [kmer_index.get(sequence[i:i+k]) for i in range(len(sequence) - k + 1) if sequence[i:i+k] in kmer_index]

# Assuming max_seq_length was determined during training and is known here
max_seq_length = 21690  # Update this based on the actual max length from training

def parse_fasta_and_vectorize(file_path, k, kmer_index, max_seq_length):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq).upper().replace('\n', '').replace(' ', '')
        kmer_ids = sequence_to_kmer_ids(sequence, k, kmer_index)
        sequences.append(kmer_ids)
    # Pad sequences to have uniform length
    sequences_padded = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype=np.int16)
    return sequences_padded

# Load the trained model
model_path = '/home/SharmaLab/mark_CNN/results/exon_CNN_dishuffle_negs_7mer.h5'
model = load_model(model_path)

# Set k length
k = 7
kmer_index = build_kmer_index(k)

# Define paths for positive and negative samples
positive_file = '/path/to/exons_from_1000_mouse_transcripts.fa'  # Update this with the actual path
negative_file = '/path/to/mark_CNN/ag_to_gt_sequences_from_intergenic.fa'  # Update this with the actual path

# Parse, vectorize, and pad positive and negative samples
positive_sequences_padded = parse_fasta_and_vectorize(positive_file, k, kmer_index, max_seq_length)
negative_sequences_padded = parse_fasta_and_vectorize(negative_file, k, kmer_index, max_seq_length)

# Prepare labels
positive_labels = np.ones(len(positive_sequences_padded))
negative_labels = np.zeros(len(negative_sequences_padded))

# Combine data and labels
X_test = np.concatenate((positive_sequences_padded, negative_sequences_padded))
y_test = np.concatenate((positive_labels, negative_labels))

# Test the model
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int).flatten()

# Evaluate performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("F1 Score:", f1_score(y_test, y_pred_binary))
