import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from Bio import SeqIO
from sklearn.model_selection import train_test_split, KFold
import itertools

np.random.seed(42)
tf.random.set_seed(42)

def parse_fasta(file_path, label):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq).upper().replace('\n', '').replace(' ', '')
        sequences.append(sequence)
    return sequences, [label] * len(sequences)

def kmer_count_vectorization(sequences, k):
    all_kmers = itertools.product('ACGT', repeat=k)
    kmer_dict = {"".join(kmer): idx for idx, kmer in enumerate(all_kmers)}

    vectorized_data = []
    for sequence in sequences:
        kmer_counts = np.zeros(4**k, dtype=np.int16)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer in kmer_dict:
                kmer_counts[kmer_dict[kmer]] += 1
        vectorized_data.append(kmer_counts)
    return np.array(vectorized_data)

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=4),
        Dropout(0.4),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',
                  tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
                  tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()])
    return model

# create empty lists to store the metrics for each fold
losses = []
accuracies = []
val_losses = []
val_accuracies = []
tps = []
fps = []
tns = []
fns = []

# Set k and file paths
k = 5
true_exon_file = '/path/to/true_exon_file.fa'
false_exon_file = '/path/to/dishuffled_exon_file.fa'

#################Retraining#################
#false_exon_file = '/path/to/dishuffled_plus_fp_exon_file.fa'
############################################

# Parse FASTA files and vectorize sequences
true_sequences, true_labels = parse_fasta(true_exon_file, 1)
false_sequences, false_labels = parse_fasta(false_exon_file, 0)
all_sequences = true_sequences + false_sequences
labels = true_labels + false_labels
vectorized_sequences = kmer_count_vectorization(all_sequences, k)

# Convert labels to numpy arrays
labels = np.array(labels)
vectorized_sequences = np.array(vectorized_sequences)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
for train_index, test_index in kf.split(vectorized_sequences):
    X_train, X_test = vectorized_sequences[train_index], vectorized_sequences[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Reshape data for CNN
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Create and train model
    model = create_cnn_model((X_train.shape[1], 1))
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    print('Training for fold', fold_no, '...')
    history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, shuffle=True, callbacks=[early_stopping_callback])

    # Evaluate the model
    test_metrics = model.evaluate(X_test, y_test)
    losses.append(test_metrics[0])
    accuracies.append(test_metrics[1])
    tps.append(test_metrics[2])
    fps.append(test_metrics[3])
    tns.append(test_metrics[4])
    fns.append(test_metrics[5])
    val_losses.append(history.history['val_loss'][-1])
    val_accuracies.append(history.history['val_accuracy'][-1])

    fold_no += 1

# Calculate and print the average and standard deviation of the metrics
mean_loss = np.mean(losses)
std_loss = np.std(losses)
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)
mean_val_accuracy = np.mean(val_accuracies)
std_val_accuracy = np.std(val_accuracies)
mean_tp = np.mean(tps)
std_tp = np.std(tps)
mean_fp = np.mean(fps)
std_fp = np.std(fps)
mean_tn = np.mean(tns)
std_tn = np.std(tns)
mean_fn = np.mean(fns)
std_fn = np.std(fns)

precision = mean_tp / (mean_tp + mean_fp)
recall = mean_tp / (mean_tp + mean_fn)
specificity = mean_tn / (mean_tn + mean_fp)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Mean loss: {mean_loss:.4f} +/- {std_loss:.4f}")
print(f"Mean accuracy: {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
print(f"Mean validation loss: {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
print(f"Mean validation accuracy: {mean_val_accuracy:.4f} +/- {std_val_accuracy:.4f}")
print(f"Mean true positives: {mean_tp:.4f} +/- {std_tp:.4f}")
print(f"Mean false positives: {mean_fp:.4f} +/- {std_fp:.4f}")
print(f"Mean true negatives: {mean_tn:.4f} +/- {std_tn:.4f}")
print(f"Mean false negatives: {mean_fn:.4f} +/- {std_fn:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 score: {f1_score:.4f}")
