import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

np.random.seed(42)
random.seed(42)

def parse_true(file_name):
    sequences = []
    current_sequence = ""
    with open(file_name, "r") as f:
        for line in f:
            if line[0] == ">":
                if current_sequence != "":
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line.strip()
        if current_sequence != "":
            sequences.append(current_sequence)
    return sequences

# Using the BED files of splice site sequences, these were converted to FASTA files using twoBitToFa, then the
# donor FASTAs (pos and neg strand) were combined, and the acceptor FASTAs (pos and neg strand) were combined.

acceptor_seq_len1 = 210

fasta_file_donor_plus_pos_strand = f"/home/mark_bromell/Human/human_donor_plus_minus_{acceptor_seq_len1}_pos_strand_thesis.fa"
sequences1 = parse_true(fasta_file_donor_plus_pos_strand)

fasta_file_donor_minus_neg_strand = f"/home/mark_bromell/Human/human_donor_plus_minus_{acceptor_seq_len1}_neg_strand_thesis_rev_comp.fa"
sequences2 = parse_true(fasta_file_donor_minus_neg_strand)

seqs1 = sequences1 + sequences2

#############################uncomment when processing acceptors#############################
# fasta_file_acceptor_minus_pos_strand = f"/home/mark/other_scripts/human_acceptor_plus_minus_{acceptor_seq_len1}_pos_strand_thesis.fa"
# sequences3 = parse_true(fasta_file_acceptor_minus_pos_strand)
# # print(len(sequences3))
# print(f"True acceptor sequences in {fasta_file_acceptor_minus_pos_strand}: {sequences3[:6]} etc.")

# fasta_file_acceptor_plus_neg_strand = f"/home/mark/other_scripts/human_acceptor_plus_minus_{acceptor_seq_len1}_neg_strand_thesis_rev_comp.fa"
# sequences4 = parse_true(fasta_file_acceptor_plus_neg_strand)
# # print(len(sequences3))
# print(f"True acceptor sequences in {fasta_file_acceptor_plus_neg_strand}: {sequences4[:6]} etc.")

# seqs1 = sequences3 + sequences4
#############################################################################################

n = 1.0

len_of_n_percent = len(seqs1) * n

seqs1 = random.sample(seqs1, int(len(seqs1) * n))
print(f"len true pos list equal to len_of_n_percent: {len(seqs1)}")

true_pos_dict = {item: 1 for item in seqs1}
print(f"final len of pos dict{len(true_pos_dict)}")

#===========================================================================================#

def get_indices(input_seq, sub_str, upstream_length, downstream_length, target_count):
    str_len = len(input_seq)
    substr_len = len(sub_str)
    i = upstream_length  # Using upstream_length as the starting point
    count = 0

    while i < (str_len - substr_len - downstream_length) and count < target_count:
        if input_seq[i:i + substr_len] == sub_str:
            yield i
            i += upstream_length + substr_len + downstream_length  # Jumping over the entire extracted sequence
            count += 1
        i += 1

def process_file(FI, counts, n_gt, n_gc, intergenic_sequence_dict, upstream_length, downstream_length):
    duplicate_count = 0
    for line in FI:
        _, seq = line.rstrip('\n').split('\t')
        seq = seq.upper()

        for dinuc in counts.keys():
            target_count = n_gt if dinuc == 'GT' else n_gc 
            pos_list = list(get_indices(seq, dinuc, upstream_length, downstream_length, target_count - counts[dinuc]))
            pos_list = [pos for pos in pos_list if pos - upstream_length >= 0 and pos + downstream_length + len(dinuc) <= len(seq)]
            
            for pos in pos_list:
                extracted_sequence = seq[pos-upstream_length:pos+downstream_length+len(dinuc)]
                if extracted_sequence in intergenic_sequence_dict:
                    duplicate_count += 1
                intergenic_sequence_dict[extracted_sequence] = 0
                counts[dinuc] += 1

    return duplicate_count

if __name__ == '__main__':
    intergenic_file = '/home/mark_bromell/Human/intergenic_regions.txt'

    # Example lengths
    upstream_length = acceptor_seq_len1 
    downstream_length = acceptor_seq_len1
    
    total_count = len(true_pos_dict) * 3
    n_gt = int(0.9 * total_count)
    n_gc = total_count - n_gt
    
    counts = {'GT': 0, 'GC': 0}
    intergenic_sequence_dict = {}
    duplicate_count = 0 

    # Process intergenic regions
    with open(intergenic_file, 'r') as FI:
        duplicate_count += process_file(FI, counts, n_gt, n_gc, intergenic_sequence_dict, upstream_length, downstream_length)

    # Print or further process the intergenic_sequence_dict as needed
    print(len(intergenic_sequence_dict))
    print(f"Number of duplicate sequences encountered: {duplicate_count}")

for i, (key, value) in enumerate(intergenic_sequence_dict.items()):
    print(f"{key}: {value}")
    if i == 3:
        break

#############################uncomment when processing acceptors#############################

# def get_indices(input_seq, sub_str, upstream_length, downstream_length, target_count):
#     str_len = len(input_seq)
#     substr_len = len(sub_str)
#     i = upstream_length  # Use upstream_length as the starting point
#     count = 0

#     while i < (str_len - substr_len - downstream_length) and count < target_count:
#         if input_seq[i:i + substr_len] == sub_str:
#             yield i
#             i += upstream_length + substr_len + downstream_length  # Jump over the entire extracted sequence
#             count += 1
#         i += 1

# def process_file(FI, counts, n_ag, intergenic_sequence_dict, upstream_length, downstream_length):
#     duplicate_count = 0
#     for line in FI:
#         _, seq = line.rstrip('\n').split('\t')
#         seq = seq.upper()

#         for dinuc in counts.keys():
#             target_count = n_ag
#             pos_list = list(get_indices(seq, dinuc, upstream_length, downstream_length, target_count - counts[dinuc]))
#             pos_list = [pos for pos in pos_list if pos - upstream_length >= 0 and pos + downstream_length + len(dinuc) <= len(seq)]
            
#             for pos in pos_list:
#                 extracted_sequence = seq[pos-upstream_length:pos+downstream_length+len(dinuc)]
#                 if extracted_sequence in intergenic_sequence_dict:
#                     duplicate_count += 1
#                 intergenic_sequence_dict[extracted_sequence] = 0
#                 counts[dinuc] += 1

#     return duplicate_count

# if __name__ == '__main__':
#     intergenic_file = '/path/to/intergenic_regions.txt'

#     # Example lengths
#     upstream_length = acceptor_seq_len1  
#     downstream_length = acceptor_seq_len1  
    
#     total_count = len(true_pos_dict) * 3
#     n_ag = int(1 * total_count)
    
#     counts = {'AG': 0}
#     intergenic_sequence_dict = {}
#     duplicate_count = 0

#     # Process intergenic regions
#     with open(intergenic_file, 'r') as FI:
#         duplicate_count += process_file(FI, counts, n_ag, intergenic_sequence_dict, upstream_length, downstream_length)

#     # Print or further process the intergenic_sequence_dict as needed
#     print(len(intergenic_sequence_dict))
#     print(f"Number of duplicate sequences encountered: {duplicate_count}")

# for i, (key, value) in enumerate(intergenic_sequence_dict.items()):
#     print(f"{key}: {value}")
#     if i == 3:
#         break

#############################################################################################

#===========================================================================================#

true_pos_series = pd.Series(true_pos_dict).reset_index()
true_pos_series.columns = ['key', 'class']
true_pos_series.index = true_pos_series.index + 1

true_neg_series = pd.Series(intergenic_sequence_dict).reset_index()
true_neg_series.columns = ['key', 'class']
true_neg_series.index = true_neg_series.index + 1

df2 = pd.concat([true_pos_series, true_neg_series], ignore_index=True)

train_samples = df2['key'].tolist()
train_labels = df2['class'].tolist()

def filter_sequences(train_samples, train_labels):
    # Ensure train_labels is a numpy array
    train_labels_array = np.array(train_labels)
    
    # Step 1: Get indices of sequences that do not contain 'N' or 'n'
    valid_indices = [i for i, seq in enumerate(train_samples) if 'N' not in seq and 'n' not in seq]

    # Step 2: Filter out sequences and labels using the valid indices
    filtered_samples = [train_samples[i] for i in valid_indices]
    filtered_labels = train_labels_array[valid_indices].tolist()

    return filtered_samples, filtered_labels

filtered_samples, filtered_labels = filter_sequences(train_samples, train_labels)

train_labels, train_samples = shuffle(filtered_labels, filtered_samples, random_state=42)

print(train_samples[0:11])
print(train_labels[0:11])

nucleotide_dict = {
    "A": [1, 0, 0, 0],
    "G": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "a": [1, 0, 0, 0],
    "g": [0, 1, 0, 0],
    "c": [0, 0, 1, 0],
    "t": [0, 0, 0, 1],
    "n": [0.25, 0.25, 0.25, 0.25],
    "N": [0.25, 0.25, 0.25, 0.25]
}

def one_hot_encode_list(seq_list, nucleotide_dict):
    max_len = max([len(seq) for seq in seq_list])
    seq_array = np.zeros((len(seq_list), max_len, 4))
    for i, seq in enumerate(seq_list):
        for j, nt in enumerate(seq):
            if nt in nucleotide_dict:
                if nt.isupper():
                    seq_array[i, j, :] = nucleotide_dict[nt]
                else:
                    seq_array[i, j, :] = nucleotide_dict[nt.upper()]
    return seq_array


one_hot_train_samples = one_hot_encode_list(train_samples, nucleotide_dict)
one_hot_train_samples = one_hot_train_samples.astype('int16')

train_labels = np.array(list(train_labels))
train_labels = train_labels.astype('int16')

np.save(f'donor_one_hot_train_samples_{acceptor_seq_len1}nt_intergenicnegs_total_ssmiddle.npy', one_hot_train_samples)
np.save(f'donor_train_labels_{acceptor_seq_len1}nt_intergenicnegs_total_ssmiddle.npy', train_labels)
