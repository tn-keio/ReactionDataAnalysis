# Compact-bottleneck autoencoder baselines (reconstruction error as the anomaly score).
#
# DenseAE       : dense 34-16-z-16-34 (ReLU hidden layers, linear code, linear output).
# ConvCompactAE : the topology of AutoencoderModel (model_autoencoder.py: Conv16-pool-Conv8-pool and
#                 the mirrored decoder) with a linear Dense(z) code inserted between encoder and
#                 decoder, and a linear output.
# Both use the optimiser, loss, epochs, batch size and early stopping of AutoencoderModel.  Inputs are
# not clipped, so the outputs are linear.

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping


class DenseAE:
    def __init__(self, input_length, z, hidden=16, lr=1e-3):
        inp = layers.Input(shape=(input_length,))
        x = layers.Dense(hidden, activation='relu')(inp)
        x = layers.Dense(z, activation=None, name='bottleneck')(x)      # linear code (no dead units at small z)
        x = layers.Dense(hidden, activation='relu')(x)
        out = layers.Dense(input_length, activation=None)(x)
        self.model = Model(inp, out)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    def train(self, X_tr, X_val, epochs=50, batch_size=32):
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = self.model.fit(X_tr, X_tr, epochs=epochs, batch_size=batch_size, shuffle=True,
                           validation_data=(X_val, X_val), callbacks=[es], verbose=0)
        return dict(best_val_mse=float(np.min(h.history['val_loss'])), epochs=len(h.history['val_loss']))

    def score(self, X):
        return np.mean((X - self.model.predict(X, batch_size=1024, verbose=0)) ** 2, axis=1)


class ConvCompactAE:
    def __init__(self, input_length, z, lr=1e-3):
        L = input_length
        enc_len = int(math.ceil(math.ceil(L / 2) / 2))          # 34 -> 9
        inp = layers.Input(shape=(L, 1))
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(inp)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Flatten()(x)
        code = layers.Dense(z, activation=None, name='bottleneck')(x)
        x = layers.Dense(enc_len * 8, activation='relu')(code)
        x = layers.Reshape((enc_len, 8))(x)
        x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(1, 3, activation=None, padding='same')(x)
        out_len = x.shape[1]
        if out_len is not None and out_len > L:
            d = out_len - L
            x = layers.Cropping1D((d // 2, d - d // 2))(x)
        self.model = Model(inp, x)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    def train(self, X_tr, X_val, epochs=50, batch_size=32):
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = self.model.fit(X_tr[..., None], X_tr[..., None], epochs=epochs, batch_size=batch_size,
                           shuffle=True, validation_data=(X_val[..., None], X_val[..., None]),
                           callbacks=[es], verbose=0)
        return dict(best_val_mse=float(np.min(h.history['val_loss'])), epochs=len(h.history['val_loss']))

    def score(self, X):
        p = self.model.predict(X[..., None], batch_size=1024, verbose=0)[..., 0]
        return np.mean((X - p) ** 2, axis=1)
