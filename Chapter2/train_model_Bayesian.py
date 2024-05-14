# Bayesian optimization gave us an idea of the best hyperparameters and and architecture to use

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from keras_tuner.engine import objective as obj_module
from keras_tuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
import os


np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

data1 = np.load(f'/path/to/train_data.npy')

train_samples = data1
train_samples = train_samples.astype(np.float32)

data2 = np.load(f'/path/to/train_labels.npy')

train_labels = data2
train_labels = train_labels.astype(np.float32)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def build_model(hp):
    model = keras.Sequential()
    
    # Define hyperparameters for the number of conv1D layers
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=5, step=1)
    
    # Add conv1D and optional maxpool1D layers
    for i in range(num_conv_layers):
        filters = hp.Int(f'filters_{i}', min_value=16, max_value=128, step=16)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7, 9])
        
        # L1 & L2 regularization
        l1_value = hp.Float(f'l1_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-4)
        l2_value = hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-4)
        
        model.add(keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                                      kernel_regularizer=l1_l2(l1=l1_value, l2=l2_value)))
        
        # Batch Normalization
        if hp.Choice(f'batchnorm_{i}', values=[0, 1]):
            model.add(BatchNormalization())
        
        # Dropout
        if hp.Choice(f'dropout_{i}', values=[0, 1]):
            dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(rate=dropout_rate))
        
        # MaxPooling
        use_maxpool = hp.Choice(f'use_maxpool_layer_{i}', values=[0, 1])
        if use_maxpool:
            pool_size = hp.Choice(f'pool_size_{i}', values=[2, 3, 4])
            model.add(keras.layers.MaxPooling1D(pool_size=pool_size))
            
    # Flatten the output from the convolutions
    model.add(keras.layers.Flatten())

    # Add dense layers
    num_dense_layers = hp.Int('num_dense_layers', min_value=0, max_value=4, step=1)
    for i in range(num_dense_layers):
        units = hp.Int(f'nodes_{i}', min_value=1, max_value=300, step=6)
        activations = hp.Choice(f'activation_{i}', ['relu', 'sigmoid', 'tanh'])
        
        # L1 & L2 regularization for dense layers
        l1_value_dense = hp.Float(f'dense_l1_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-4)
        l2_value_dense = hp.Float(f'dense_l2_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-4)
        
        model.add(keras.layers.Dense(units, activation=activations, 
                                     kernel_regularizer=l1_l2(l1=l1_value_dense, l2=l2_value_dense)))
        
        # Dropout for dense layers
        if hp.Choice(f'dense_dropout_{i}', values=[0, 1]):
            dense_dropout_rate = hp.Float(f'dense_dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(rate=dense_dropout_rate))

    # Add the output layer with sigmoid activation
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model with binary cross-entropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives(), f1_score])
    return model

custom_objective = obj_module.Objective('val_f1_score', 'max')

tuner = BayesianOptimization(
    build_model, 
    objective=custom_objective, 
    max_trials=50, 
    directory='/home/mark/saved_models/Baye_results', 
    project_name=f'Bayesian_model'
)


tuner.search_space_summary()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        val_f1_score = logs.get('val_f1_score')
        
        if val_accuracy is not None and val_accuracy <= 0.8:
            print("Stopping training due to val_accuracy <= 0.8")
            self.model.stop_training = True
        elif val_f1_score is not None and val_f1_score <= 0.8:
            print("Stopping training due to val_f1_score <= 0.8")
            self.model.stop_training = True

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=2)

# Split the data into training and validation sets for this fold
X_train, X_val, y_train, y_val = train_test_split(train_samples, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32, callbacks=[stop_early, CustomCallback()])

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('/path/to/Bayesian_model.h5')
print(best_model.summary())
