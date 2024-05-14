import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import gc
import argparse

# Sequences to be tested should be 400 nucleotides long and their corresponding labels should match up to each
# sequence in the sequences file. CNN.h5, acceptor_dis.h5 and donor_dis.h5 can all be found at the SpliceFinder
# github

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, required=True)
parser.add_argument("--end_idx", type=int, required=True)
args = parser.parse_args()

# File paths
sequences_path = '400_nuc_sequences.txt'
labels_path = 'labels.txt'
output_file_path = 'output_scores.txt'
error_log_path = 'error_log.txt'

# Load models
model = load_model('CNN.h5')
acceptor_model = load_model('acceptor_dis.h5')
donor_model = load_model('donor_dis.h5')

# Function to process a batch of data
def process_batch(batch_sequences, batch_labels, start_idx):
    x_test = np.array(batch_sequences, dtype=np.float32).reshape(-1, 400, 4)
    y_test = to_categorical(np.array(batch_labels, dtype=int), num_classes=3)

    predictions = model.predict(x_test, batch_size=256, verbose=0)
    predict = np.argmax(predictions, axis=1)

    unscored_indices = []  # List to track indices of unscored sequences

    test_defined = False
    for i in range(len(predict)):
        if predict[i] != 2:
            test = np.expand_dims(x_test[i], axis=0)
            test_defined = True
            if predict[i] == 0:
                check = np.argmax(acceptor_model.predict(test, verbose=0), axis=1)
                if check[0] == 1:
                    predict[i] = 2
            elif predict[i] == 1:
                check = np.argmax(donor_model.predict(test, verbose=0), axis=1)
                if check[0] == 1:
                    predict[i] = 2
        if predict[i] not in [0, 1, 2]:
            unscored_indices.append(start_idx + i)

    with open(output_file_path, 'a') as file:
        for score in predict:
            file.write(str(score) + '\n')

    if unscored_indices:
        with open(error_log_path, 'a') as error_file:
            for idx in unscored_indices:
                error_file.write(f'Unscored sequence at index: {idx}\n')

    del x_test, y_test, predictions, predict
    if test_defined:
        del test
    gc.collect()

# Function to read data in batches
def read_data_in_batches(start_idx, end_idx, batch_size):
    current_idx = 0
    batch_sequences = []
    batch_labels = []

    with open(sequences_path, 'r') as seq_file, open(labels_path, 'r') as label_file:
        while True:
            # Read the sequence line and its corresponding label line
            seq_line = seq_file.readline()
            label_line = label_file.readline()

            # Break if either sequence or label file has reached the end
            if not seq_line or not label_line:
                if batch_sequences:
                    yield batch_sequences, batch_labels
                break

            # Increment current index and skip the header (every odd line)
            current_idx += 1
            if current_idx % 2 == 1:
                continue

            # Process only the even lines (one-hot encoded sequences)
            if current_idx // 2 >= start_idx and current_idx // 2 < end_idx:
                batch_sequences.append([float(num) for num in seq_line.split()])
                batch_labels.append(int(label_line.strip()))

            # Yield a batch when it's full or when the end index is reached
            if len(batch_sequences) == batch_size or (current_idx // 2 >= end_idx and batch_sequences):
                yield batch_sequences, batch_labels
                batch_sequences = []
                batch_labels = []

# Batch processing
batch_size = 100
for batch_sequences, batch_labels in read_data_in_batches(args.start_idx, args.end_idx, batch_size):
    print(f"Processing sequences from {args.start_idx} to {args.end_idx}")
    process_batch(batch_sequences, batch_labels, args.start_idx)
