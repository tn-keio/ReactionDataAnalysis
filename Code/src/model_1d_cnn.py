# Abnormal Reaction Detection System
# Copyright (C) 2026
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D,
    BatchNormalization, GlobalAveragePooling1D, Dense, Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class CNN1DFeatureExtractor:
    """
    1D-CNN feature extractor for Phase 2 anomaly detection.

    Architecture differs from Phase 1 (AutoencoderModel) in two key ways:
    - Deeper encoder: 3 Conv blocks (32→64→128 filters) vs. Phase 1's 2 blocks (16→8)
    - Feature-based anomaly detection: encoder output fed to Isolation Forest,
      rather than using reconstruction error as the anomaly signal.

    Training: autoencoder pretraining on normal data (unsupervised).
    Inference: encoder extracts a 64-dim feature vector per sample.
    """

    def __init__(self, input_length=28, feature_dim=64):
        self.input_length = input_length
        self.feature_dim = feature_dim
        self._encoder = None
        self._autoencoder = None
        self._build_models()
        self._autoencoder.compile(optimizer='adam', loss='mse')

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self):
        inp = Input(shape=(self.input_length, 1), name='input')

        # --- Encoder ---
        x = Conv1D(32, 5, padding='same', name='enc_conv1')(inp)
        x = BatchNormalization(name='enc_bn1')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same', name='enc_pool1')(x)  # -> (14, 32)

        x = Conv1D(64, 3, padding='same', name='enc_conv2')(x)
        x = BatchNormalization(name='enc_bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same', name='enc_pool2')(x)  # -> (7, 64)

        x = Conv1D(128, 3, padding='same', name='enc_conv3')(x)
        x = BatchNormalization(name='enc_bn3')(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling1D(name='gap')(x)               # -> (128,)
        features = Dense(self.feature_dim, activation='relu', name='features')(x)

        self._encoder = Model(inp, features, name='encoder')

        # --- Decoder (for autoencoder pretraining only) ---
        # Mirror structure: Dense -> reshape -> Conv1DTranspose via UpSampling
        encoded_inp = self._encoder.output

        # Length-aware bottleneck so the decoder reconstructs to ANY input_length (was hard-coded to 7
        # = correct only for L=28).  The encoder halves the length twice (two MaxPooling1D(2,'same')),
        # so the pre-GAP length is enc_len = ceil(ceil(L/2)/2).  L=28 -> 7 (IDENTICAL to the original,
        # so 11-38 behaviour is unchanged); L=34 -> 9.  The Cropping1D below trims 4*enc_len back to L.
        enc_len = int(np.ceil(np.ceil(self.input_length / 2) / 2))
        x = Dense(enc_len * 128, activation='relu', name='dec_dense')(encoded_inp)
        x = tf.keras.layers.Reshape((enc_len, 128), name='dec_reshape')(x)

        x = Conv1D(128, 3, padding='same', name='dec_conv1')(x)
        x = BatchNormalization(name='dec_bn1')(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2, name='dec_up1')(x)                  # -> (14, 128)

        x = Conv1D(64, 3, padding='same', name='dec_conv2')(x)
        x = BatchNormalization(name='dec_bn2')(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2, name='dec_up2')(x)                  # -> (28, 64)

        x = Conv1D(32, 5, padding='same', name='dec_conv3')(x)
        x = BatchNormalization(name='dec_bn3')(x)
        x = Activation('relu')(x)

        # linear output: inputs are not clipped, so targets can lie outside [0, 1]
        decoded = Conv1D(1, 3, padding='same', activation=None, name='output')(x)

        # Crop if upsampling produced extra length
        final_len = decoded.shape[1]
        if final_len is not None and final_len > self.input_length:
            diff = final_len - self.input_length
            crop_start = diff // 2
            crop_end = diff - crop_start
            decoded = Cropping1D((crop_start, crop_end), name='crop')(decoded)

        self._autoencoder = Model(inp, decoded, name='cnn1d_autoencoder')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, X_val, epochs=50, batch_size=32,
              save_path='best_model_1d_cnn.keras'):
        checkpoint = ModelCheckpoint(
            save_path, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min'
        )
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        history = self._autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[checkpoint, early_stop],
            verbose=2
        )
        return history

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, X):
        """Return encoder features: shape (N, feature_dim)."""
        return self._encoder.predict(X, verbose=0)

    def reconstruct(self, X):
        """Return autoencoder reconstruction (for diagnostics)."""
        return self._autoencoder.predict(X, verbose=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        self._autoencoder.save(path)

    @classmethod
    def from_saved(cls, path, feature_dim=64):
        """Load a pretrained autoencoder and wire up the encoder."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        instance = cls.__new__(cls)
        instance._autoencoder = load_model(path)
        instance.input_length = instance._autoencoder.input_shape[1]
        instance.feature_dim = feature_dim
        # Re-derive encoder from loaded autoencoder weights
        instance._encoder = Model(
            instance._autoencoder.input,
            instance._autoencoder.get_layer('features').output,
            name='encoder'
        )
        return instance

    def summary(self):
        print("=== Encoder ===")
        self._encoder.summary()
        print("\n=== Full Autoencoder ===")
        self._autoencoder.summary()
