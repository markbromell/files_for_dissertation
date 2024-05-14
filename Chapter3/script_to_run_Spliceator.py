import tensorflow as tf
import os
import sys
import argparse
from os.path import basename
from pathlib import Path
import numpy as np
from tqdm import tqdm

from Bio import SeqIO


model_path = os.getcwd() + "/Models/"

# Check the version of Tensorflow
def check_version():
    print("[INFO] : TF Version: ", tf.__version__)

    tpr = str(tf.__version__).split(".")
    version = int(tpr[0])
    release = int(tpr[1])

    if tf.__version__ != "2.4.1":
        if version < 2:
            print(f"[ERROR] please use the version 2 of Tensorflow")
            exit()
        if version == 2 and release < 4:
            print(f"Your version of TensorFlow ({tf.__version__}), please upgrade to version 2.4.1 (or use the virtual environment)")
            print("[INFO] We cannot guarantee that the program will work with this version.\nHowever, it is possible to use the virtual environment available in git to have a stable version of Spliceator\n")

    else:
        print("[INFO] We cannot guarantee that the program will work in the next Tensorflow update, but we will do what is necessary.\nHowever, it is possible to use the virtual environment available in git to have a stable version of Spliceator")

def main():
    print("===================================\n")
    print("         SPLICEATOR v1.0\n")
    print("===================================\n")
    print("Thank to use it\n")
    parser = argparse.ArgumentParser()

    check_version()

    # REQUIRED
    parser.add_argument('-f', '--fasta', required=True, help="Raw DNA sequence file (fasta format) to be analysed")
    # OPTIONAL
    #parser.add_argument('-v', '--version', version=' 2.0')
    parser.add_argument('-md', '--model_donor', default=200, type=int,    choices=[20, 80, 140, 200, 400, 600], help="Length of input sequences for Donor model , default 200")
    parser.add_argument('-ma', '--model_acceptor', default=200, type=int, choices=[20, 80, 140, 200, 400, 600], help="Length of input sequences for Acceptor model, default 140")
    parser.add_argument('-ta', '--threshold_donor', default=98, type=int, help="Reliability of the predictions [between 50-100] for donor model")
    parser.add_argument('-td', '--threshold_acceptor', default=98, type=int, help="Reliability of the predictions [between 50-100] for acceptor model")
    parser.add_argument('-o',  '--output', default="", help="Output file with the prediction values, preferably with the .csv extension")
    parser.add_argument('-a',  '--all', default=False, type=bool, choices=[True, False], help="To obtain the score for each position (nucleotide), default=False" )
    args = parser.parse_args()

    model_donor_size = int(args.model_donor)
    model_acceptor_size = int(args.model_acceptor)
    reliability_don = int(args.threshold_donor)
    reliability_acc = int(args.threshold_acceptor)
    output = args.output
    all_results = args.all
    sequence = ""

    if reliability_don < 50 or reliability_acc < 50:
        print('[ERROR] td or ta (Reliability) need to be > 50')
        exit()

    # Writing the header in the output file
    if output != "":
        file = open(output, "a")
        header = "ID;#;SS_type;Sequence;Position;Score\n"
        file.write(header)
        file.close()

    fileName, fileExtension = os.path.splitext(args.fasta)

    # Reading the input file
    if fileExtension == ".fasta" or fileExtension == ".txt" or fileExtension == ".fa":
        print(f"[INFO] File {args.fasta} is loaded")

        liste_sequence = []

        # Check if the file is a fasta file
        validation=False
        with open(args.fasta, "r") as file_R1:
            for ligne in file_R1:
                if ">" in ligne:
                    validation=True

        if not validation:
            print(f"[ERROR] {args.fasta} is not a fasta file")
            exit()

        all_header = []
        for record in SeqIO.parse(args.fasta, "fasta"):
            print(record)
            liste_sequence.append(str(record.seq))
            all_header.append(str(record.id))

        # loading models
        model_donor    = tf.keras.models.load_model(model_path + f"donor_{model_donor_size}.h5" )
        model_acceptor = tf.keras.models.load_model(model_path + f"acceptor_{model_acceptor_size}.h5")
        print(f"[INFO] Model Donor_{model_donor_size} and model Acceptor_{model_acceptor_size} are correctly loaded")

        if output == "":
            print("\n== RESULTS ==\n")

        for i, sequence in enumerate(liste_sequence):
            # Evaluation of all sequences in fasta file
            dico_donor = evaluate(sequence, model_donor, all_results, threshold=reliability_don, window_size=model_donor_size)
            dico_acceptor = evaluate(sequence, model_acceptor, all_results, threshold=reliability_acc, window_size=model_acceptor_size)

            dico_donor, dico_acceptor = evaluate(sequence, model_donor, model_acceptor, all_results, threshold=reliability_don, window_size=model_donor_size)

            compteur_donor = 0
            compteur_acceptor = 0

            for k, v in dico_donor.items():
                compteur_donor += 1
                x = int(model_donor_size/2)
                position = int(k) + x
                seq = v[1]
                proba = round(float(v[0][1]),3)
                ligne = f'{all_header[i]};#{compteur_donor};Donor;{seq.replace("N","")};{position-int(model_donor_size/2)+1};{proba}\n'

                if output == "":
                    print(ligne.strip().replace(";"," "))
                else:
                    file = open(output, "a")
                    file.write(ligne)
                    file.close()

            for k, v in dico_acceptor.items():
                compteur_acceptor += 1
                x = int(model_acceptor_size/2)
                position = int(k) + x
                seq = v[1]
                proba = round(float(v[0][1]),3)
                ligne = f'{all_header[i]};#{compteur_acceptor};Acceptor;{seq.replace("N","")};{position-int(model_acceptor_size/2)+1};{proba}\n'

                if output == "":
                    print(ligne.strip().replace(";"," "))
                else:
                    file = open(output, "a")
                    file.write(ligne)
                    file.close()

        if output != "":
            print(f"[INFO] {output} is written")

    else:
        print("[ERROR] The file extension is not .fasta or .txt")
        exit()

def one_hot_encoding(sequence):

    sequence = sequence.upper()
    encoded_sequence = ""

    for nuc in sequence:
        if nuc == "A":
            encoded_sequence += "1    0    0    0    "
        elif nuc == "C":
            encoded_sequence += "0    1    0    0    "
        elif nuc == "G":
            encoded_sequence += "0    0    1    0    "
        elif nuc == "T":
            encoded_sequence += "0    0    0    1    "
        elif nuc == "N":
            encoded_sequence += "0    0    0    0    "
        else:
            encoded_sequence = ""
            break

    return encoded_sequence

def find_seq(sequence, pos, size=400):
    sequence = sequence.upper()
    longeur_seq = len(sequence)

    for i in range(longeur_seq):

        if i == pos:
            if size == 20:
                window = sequence[i:i+20]
            elif size == 80:
                window = sequence[i+30:i+50]
            elif size == 140:
                window = sequence[i+60:i+80]
            elif size == 200:
                window = sequence[i+90:i+110]
            elif size == 400:
                window = sequence[i+190:i+210]

            return window

# Returns the positions of the donor and acceptor sites
def evaluate(sequence, model_donor, model_acceptor, all_results, threshold=95, window_size=400):
    # all predictions
    dico_ss_donor = {}
    dico_ss_acceptor = {}

    if all_results:
        threshold = 0
    else:
        threshold = float(threshold/100)
    # sequence to analyze
    sequence = "N"*int(window_size/2) + sequence.upper() + "N"*int(window_size/2)
    # all windowed_sequences from the raw sequences
    output_donor = []
    output_acceptor = []

    d_window_size = int(window_size/4)


    # analyze of all sequence
    for i in tqdm(range(len(sequence))):
        # windowed_sequence's size is the same size as seq_train/seq_test
        windowed_sequence = sequence[i: window_size+i]

        # Encoding the target windowed sequence in one-hot
        encoded_windowed_sequence = one_hot_encoding(windowed_sequence).replace(" ","")

        # Check the lenght of sequences, only sequences with length=window_size are analyzed
        if len(encoded_windowed_sequence) != window_size*4:
            pass
        else:
            # Conversion
            to_add = np.array(list(encoded_windowed_sequence), dtype=int)
            # Reshaping
            to_add2 = to_add.reshape(-1, int(window_size), 4)
            # Run the prediction on all windowed_sequences
            output_donor.append(model_donor.predict(to_add2))
            output_acceptor.append(model_acceptor.predict(to_add2))

    # remove last N
    del output_donor[-1]
    del output_acceptor[-1]

    # Donor
    for position, proba in enumerate(output_donor):
        proba = [proba[0][0], proba[0][1]]

        if all_results:
            s = find_seq(sequence, position, window_size)
            dico_ss_donor[str(position)] = [proba, s]
        else:
            # get predicted SS
            if proba[1] > proba[0]:
                if proba[1] >= float(threshold):
                    s = find_seq(sequence, position, window_size)

                    dico_ss_donor[str(position)] = [proba, s]

    # Acceptor
    for position, proba in enumerate(output_acceptor):
        proba = [proba[0][0], proba[0][1]]

        if all_results:
            s = find_seq(sequence, position, window_size)
            dico_ss_acceptor[str(position)] = [proba, s]
        else:
            # get predicted SS
            if proba[1] > proba[0]:
                if proba[1] >= float(threshold):
                    s = find_seq(sequence, position, window_size)

                    dico_ss_acceptor[str(position)] = [proba, s]

    return dico_ss_donor, dico_ss_acceptor


if __name__ == '__main__':
    main()
