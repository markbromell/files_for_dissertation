from __future__ import print_function
import numpy as np
import time
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import *
from sklearn import tree, metrics
from sklearn.metrics import precision_score, recall_score, classification_report, roc_auc_score
import argparse
import os
import re
from sklearn.metrics import confusion_matrix

fasta_directory = '/path/to/encoded/fasta_files'

# Information on how these fasta sequences are encoded can be found on the CNNSplice github

def load_data(sequence_file, label_file):
    y_test = np.loadtxt(label_file)
    x_test = np.loadtxt(sequence_file)
    print(x_test.shape)
    x_test = x_test.reshape(-1, 400, 4)
    y_true = y_test
    y_test = to_categorical(y_test, num_classes=2)
    return x_test, y_test


def testing_process(x_test, y_test, seq, seq_name, name, datatype="", fasta_name=""):
    model = load_model(f'./models/acceptor_cnnsplice_hs.h5')
    print(model.summary())
    loss, accuracy = model.evaluate(x_test, y_test)

    prob = model.predict(x_test)
    predicted = np.argmax(prob, axis=1)
    report = classification_report(np.argmax(y_test, axis=1), predicted, output_dict=True)

    # Save scores for each sequence in a text file
    scores_file = f"./log/{fasta_name.split('.')[0]}_scores.txt"
    with open(scores_file, 'w') as score_file:
        score_file.write("Sequence\tScore\n")
        for i, sequence_score in enumerate(prob):
            score_file.write(f"Sequence_{i + 1}\t{sequence_score[1]}\n")

    # Calculate metrics
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    class_accuracy = report['accuracy']

    # Calculate the number of true positives, true negatives, false positives and false negatives
    tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), predicted).ravel()

    # Print the results
    print("True positives:", tp)
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)

    data_metrics = {
        "fasta": fasta_name,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "class_accuracy": class_accuracy,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
    return data_metrics



def main(name, fasta_directory):
    all_results = []
    encoded_file_count = 0  # Initialize the counter

    list_name = ["hs"]  # You mentioned you only want 'hs', so I updated this list
    seq = "acceptor"
    seq_name = "acceptor"

    # Initialize the counters for the total number of true positives, true negatives, false positives and false negatives
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for fasta_name in os.listdir(fasta_directory):
        if "encoded" in fasta_name and fasta_name.endswith('.txt'):
            encoded_file_count += 1  # Increment the counter
            print(f"Processing {fasta_name}. Count: {encoded_file_count}")  # Add this line
            sequence_file = os.path.join(fasta_directory, fasta_name)
            label_file = os.path.join(fasta_directory, f"labels_{fasta_name.split('encoded_')[1]}")
            x_test, y_test = load_data(sequence_file, label_file)

            for datatype in list_name:
                metrics = testing_process(x_test, y_test, seq, seq_name, name, datatype=datatype, fasta_name=fasta_name)
                all_results.append(metrics)

                # Add the number of true positives, true negatives, false positives and false negatives to the total counters
                total_tp += metrics["tp"]
                total_tn += metrics["tn"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]

    # Saving and printing results
    with open('./log/concatenated_results_1000_3.txt', 'w') as fl:
        for result in all_results:
            fl.write(str(result) + "\n")

    for result in all_results:
        print(result)

    # Print the count of processed 'encoded' files
    print(f"Number of 'encoded' files processed: {encoded_file_count}")

    # Print the total number of true positives, true negatives, false positives and false negatives
    print("Total true positives:", total_tp)
    print("Total true negatives:", total_tn)
    print("Total false positives:", total_fp)
    print("Total false negatives:", total_fn)



def app_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, help="name of convolutional model")
    parser.add_argument("-d", "--directory", type=str, required=True, help="directory containing fasta files")
    args = parser.parse_args()
    name = args.name
    fasta_directory = args.directory
    main(name, fasta_directory)

if __name__ == '__main__':
    app_init()
