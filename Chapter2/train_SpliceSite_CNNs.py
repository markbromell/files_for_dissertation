import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

acceptor_seq_len2 = 210

data1 = np.load(f'donor_one_hot_train_samples_{acceptor_seq_len1}nt_intergenicnegs_total_ssmiddle.npy')

train_samples = data1

data2 = np.load(f'donor_train_labels_{acceptor_seq_len1}nt_intergenicnegs_total_ssmiddle.npy')

train_labels = data2

#============================================================Donor Model============================================================#

# create empty lists to store the metrics for each fold
losses = []
accuracies = []
val_losses = []
val_accuracies = []
tps = []
fps = []
tns = []
fns = []

# Define the number of folds for cross-validation
num_folds = 5

# Define the KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop over the folds
for fold, (train_indices, val_indices) in enumerate(kf.split(train_samples, train_labels)):
    print(f"Fold {fold + 1}")
    
    # Split the data into training and validation sets for this fold
    X_train, X_val, y_train, y_val = train_test_split(train_samples[train_indices], train_labels[train_indices], 
                                                      test_size=0.2, random_state=42)
    
    # Define class weights
    # class_weights = {0: 1, 1: (len(y_train)-sum(y_train))/sum(y_train)}

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=256, kernel_size=9, activation='relu', input_shape=(acceptor_seq_len2, 4)),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(88, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()])

    # Train the model on the training set for this fold
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[stop_early])

    # Evaluate the model on the validation set for this fold
    loss, accuracy, tp, fp, tn, fn = model.evaluate(train_samples[val_indices], train_labels[val_indices])
    
    # Append the metrics to the lists for this fold
    losses.append(loss)
    accuracies.append(accuracy)
    tps.append(tp)
    fps.append(fp)
    tns.append(tn)
    fns.append(fn)
    val_losses.append(history.history['val_loss'][-1])
    val_accuracies.append(history.history['val_accuracy'][-1])

#=================Save Model================#

# Path to where you want the trained model saved
model.save(f'donor_model_{acceptor_seq_len2}_nucs.h5')

#==================Metrics==================#

# Calculate the mean and standard deviation of the evaluation metrics
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

# Calculate additional evaluation metrics using the aggregated true positives, false positives, true negatives, and false negatives
precision = mean_tp / (mean_tp + mean_fp)
recall = mean_tp / (mean_tp + mean_fn)
specificity = mean_tn / (mean_tn + mean_fp)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the results
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
