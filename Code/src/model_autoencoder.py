
import os
import math
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class AutoencoderModel:
    def __init__(self, input_length=38):
        self.input_length = input_length
        self.model = self._build_model()
        self.compile()
        
    def _build_model(self):
        """Builds the Convolutional Autoencoder architecture."""
        input_layer = Input(shape=(self.input_length, 1))

        # Encoder
        x = Conv1D(16, 3, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        encoded = MaxPooling1D(2, padding='same')(x)

        # Decoder
        x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        
        # Output layer
        x = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
        
        # Dynamic Cropping
        # Calculate expected output length after pooling and upsampling
        # Pooling: ceil(L/2)
        # Upsampling: L*2
        # math is imported at module level (PEP 8)
        l1 = math.ceil(self.input_length / 2)
        l2 = math.ceil(l1 / 2)
        l3 = l2 * 2
        final_len = l3 * 2
        
        if final_len > self.input_length:
            diff = final_len - self.input_length
            crop_start = diff // 2
            crop_end = diff - crop_start
            decoded = Cropping1D(cropping=(crop_start, crop_end))(x)
        else:
            decoded = x

        model = Model(input_layer, decoded)
        return model

    def compile(self, optimizer='adam', loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def summary(self):
        self.model.summary()

    def train(self, X_train, X_val, epochs=50, batch_size=32, save_path='best_model_ae.keras'):
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[checkpoint, early_stopping],
            verbose=2
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    @classmethod
    def from_saved(cls, model_path):
        """Load a saved model without building a new architecture first."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        instance = cls.__new__(cls)
        instance.model = load_model(model_path)
        instance.input_length = instance.model.input_shape[1]
        print(f"Model loaded from {model_path}")
        return instance

    def load(self, model_path):
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self.input_length = self.model.input_shape[1]
            print(f"Model loaded from {model_path}")
        else:
            print(f"Error: Model file {model_path} not found.")

    def save(self, model_path):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
