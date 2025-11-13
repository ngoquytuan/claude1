"""
GAN (Generative Adversarial Network) for generating fake keystroke patterns
"""

import numpy as np
from typing import Dict, Any, Tuple
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not installed. Please run: pip install tensorflow")
    raise

from config.settings import (
    LATENT_DIM, BATCH_SIZE, EPOCHS_GAN, MODEL_PATH,
    GAN_LEARNING_RATE, GAN_BETA_1, GAN_DROPOUT_RATE, GAN_LEAKY_RELU_ALPHA
)
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger()


class KeystrokeGAN:
    """
    GAN for generating synthetic keystroke patterns
    """

    def __init__(self, latent_dim: int = LATENT_DIM, feature_dim: int = None):
        """
        Initialize GAN

        Args:
            latent_dim: Dimension of latent noise vector
            feature_dim: Dimension of feature vector (must match keystroke features)
        """
        if feature_dim is None:
            raise ValueError("feature_dim must be specified")

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Build combined model for training generator
        self.discriminator.trainable = False
        noise_input = keras.Input(shape=(self.latent_dim,))
        generated_features = self.generator(noise_input)
        validity = self.discriminator(generated_features)
        self.combined = Model(noise_input, validity)

        # Compile models
        optimizer_d = Adam(learning_rate=GAN_LEARNING_RATE, beta_1=GAN_BETA_1)
        optimizer_g = Adam(learning_rate=GAN_LEARNING_RATE, beta_1=GAN_BETA_1)

        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy']
        )

        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_g
        )

        self.is_trained = False

        logger.info(f"Initialized KeystrokeGAN: latent_dim={latent_dim}, feature_dim={feature_dim}")

    def build_generator(self) -> Model:
        """
        Build generator network

        Returns:
            Generator model
        """
        model = Sequential([
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=GAN_LEAKY_RELU_ALPHA),
            layers.BatchNormalization(momentum=0.8),
            layers.Dropout(GAN_DROPOUT_RATE),

            layers.Dense(512),
            layers.LeakyReLU(alpha=GAN_LEAKY_RELU_ALPHA),
            layers.BatchNormalization(momentum=0.8),
            layers.Dropout(GAN_DROPOUT_RATE),

            layers.Dense(self.feature_dim, activation='tanh')
        ], name='generator')

        logger.info("Generator built")
        return model

    def build_discriminator(self) -> Model:
        """
        Build discriminator network

        Returns:
            Discriminator model
        """
        model = Sequential([
            layers.Dense(512, input_dim=self.feature_dim),
            layers.LeakyReLU(alpha=GAN_LEAKY_RELU_ALPHA),
            layers.Dropout(GAN_DROPOUT_RATE),

            layers.Dense(256),
            layers.LeakyReLU(alpha=GAN_LEAKY_RELU_ALPHA),
            layers.Dropout(GAN_DROPOUT_RATE),

            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        logger.info("Discriminator built")
        return model

    def train(self, X_real: np.ndarray, epochs: int = EPOCHS_GAN,
              batch_size: int = BATCH_SIZE, save_interval: int = 100) -> Dict[str, list]:
        """
        Train GAN

        Args:
            X_real: Real keystroke features (n_samples, feature_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: Save checkpoint every N epochs

        Returns:
            Training history
        """
        logger.info(f"Training GAN: {len(X_real)} samples, {epochs} epochs, batch_size={batch_size}")

        # Normalize real data to [-1, 1] range (to match tanh output)
        X_real_normalized = self._normalize_data(X_real)

        # Training history
        history = {
            'd_loss': [],
            'd_acc': [],
            'g_loss': []
        }

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select random batch of real samples
            idx = np.random.randint(0, X_real_normalized.shape[0], batch_size)
            real_samples = X_real_normalized[idx]

            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)

            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train generator (wants discriminator to mistake fake as real)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Save metrics
            history['d_loss'].append(float(d_loss[0]))
            history['d_acc'].append(float(d_loss[1]))
            history['g_loss'].append(float(g_loss))

            # Print progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - D_loss: {d_loss[0]:.4f}, D_acc: {100*d_loss[1]:.2f}%, G_loss: {g_loss:.4f}")

            # Save checkpoint
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"{MODEL_PATH}gan_checkpoint_epoch_{epoch}.h5")

        self.is_trained = True
        logger.info("GAN training complete")

        return history

    def generate(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate fake keystroke patterns

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated feature vectors (n_samples, feature_dim)
        """
        if not self.is_trained:
            logger.warning("Generating from untrained model")

        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)

        # Denormalize from [-1, 1] to original scale
        # Note: This is approximate, real denormalization needs original data stats
        generated_denorm = self._denormalize_data(generated)

        return generated_denorm

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [-1, 1] range for tanh

        Args:
            data: Input data

        Returns:
            Normalized data
        """
        # Store normalization parameters
        self.data_min = np.min(data, axis=0, keepdims=True)
        self.data_max = np.max(data, axis=0, keepdims=True)

        # Avoid division by zero
        data_range = self.data_max - self.data_min
        data_range[data_range == 0] = 1.0

        # Normalize to [-1, 1]
        normalized = 2 * (data - self.data_min) / data_range - 1

        return normalized

    def _denormalize_data(self, normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize data from [-1, 1] back to original range

        Args:
            normalized: Normalized data

        Returns:
            Denormalized data
        """
        if not hasattr(self, 'data_min') or not hasattr(self, 'data_max'):
            logger.warning("Normalization parameters not found, returning as-is")
            return normalized

        data_range = self.data_max - self.data_min
        data_range[data_range == 0] = 1.0

        denormalized = (normalized + 1) / 2 * data_range + self.data_min

        return denormalized

    def save_models(self, filepath: str = None) -> str:
        """
        Save generator and discriminator models

        Args:
            filepath: Base path for saving (default: MODEL_PATH/gan_model.h5)

        Returns:
            Base path where models were saved
        """
        if filepath is None:
            ensure_dir(MODEL_PATH)
            filepath = f"{MODEL_PATH}gan_model.h5"

        base_path = filepath.replace('.h5', '')

        # Save generator
        generator_path = f"{base_path}_generator.h5"
        self.generator.save(generator_path)

        # Save discriminator
        discriminator_path = f"{base_path}_discriminator.h5"
        self.discriminator.save(discriminator_path)

        # Save normalization parameters
        import pickle
        params_path = f"{base_path}_params.pkl"
        params = {
            'latent_dim': self.latent_dim,
            'feature_dim': self.feature_dim,
            'is_trained': self.is_trained,
            'data_min': getattr(self, 'data_min', None),
            'data_max': getattr(self, 'data_max', None)
        }
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)

        logger.info(f"GAN models saved to {base_path}_*.h5")
        return base_path

    def load_models(self, filepath: str = None) -> None:
        """
        Load generator and discriminator models

        Args:
            filepath: Base path for loading (default: MODEL_PATH/gan_model.h5)
        """
        if filepath is None:
            filepath = f"{MODEL_PATH}gan_model.h5"

        base_path = filepath.replace('.h5', '')

        # Load generator
        generator_path = f"{base_path}_generator.h5"
        self.generator = keras.models.load_model(generator_path)

        # Load discriminator
        discriminator_path = f"{base_path}_discriminator.h5"
        self.discriminator = keras.models.load_model(discriminator_path)

        # Load parameters
        import pickle
        params_path = f"{base_path}_params.pkl"
        with open(params_path, 'rb') as f:
            params = pickle.load(f)

        self.latent_dim = params['latent_dim']
        self.feature_dim = params['feature_dim']
        self.is_trained = params['is_trained']
        self.data_min = params.get('data_min')
        self.data_max = params.get('data_max')

        logger.info(f"GAN models loaded from {base_path}_*.h5")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model architecture summary

        Returns:
            Dictionary with model info
        """
        return {
            'latent_dim': self.latent_dim,
            'feature_dim': self.feature_dim,
            'is_trained': self.is_trained,
            'generator_params': self.generator.count_params(),
            'discriminator_params': self.discriminator.count_params()
        }
