"""
Training script for GAN model
"""

import argparse
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from config.settings import (
    DATA_PROCESSED_PATH, MODEL_PATH, LATENT_DIM,
    EPOCHS_GAN, BATCH_SIZE, TARGET_TEXT
)
from src.data_collection.feature_extractor import FeatureExtractor
from src.models.gan import KeystrokeGAN
from src.utils.helpers import load_numpy, save_json, save_numpy, ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger('train_gan')


def check_gpu_availability():
    """
    Check and log GPU availability

    Returns:
        bool: True if GPU available, False otherwise
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            logger.info(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                logger.info(f"  [{i}] {gpu.name}")

            # Enable memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("✓ GPU memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"Could not set memory growth: {e}")

            logger.info("✓ Training will use GPU (5-10x faster)")
            return True
        else:
            logger.info("ℹ No GPU found. Training will use CPU.")
            logger.info("  Tip: For faster training, consider using GPU with CUDA")
            return False

    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        return False


def plot_training_history(history: dict, output_path: str):
    """
    Plot and save training history

    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot discriminator loss
    axes[0].plot(history['d_loss'], label='Discriminator Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Discriminator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot generator loss
    axes[1].plot(history['g_loss'], label='Generator Loss', color='orange', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Generator Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Training history plot saved to {output_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GAN model')
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to processed features (npy file)'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default=None,
        help='Path to raw data JSON (will extract features first)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS_GAN,
        help=f'Number of training epochs (default: {EPOCHS_GAN})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=LATENT_DIM,
        help=f'Latent dimension (default: {LATENT_DIM})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for model (default: data/models/gan_model.h5)'
    )

    args = parser.parse_args()

    logger.info("="*50)
    logger.info("GAN MODEL TRAINING")
    logger.info("="*50)

    # Check GPU availability
    has_gpu = check_gpu_availability()

    # Load or extract features
    if args.data:
        logger.info(f"Loading features from {args.data}")
        features = load_numpy(args.data)
    elif args.raw_data:
        logger.info(f"Extracting features from raw data: {args.raw_data}")
        extractor = FeatureExtractor()
        extractor.generate_feature_names(TARGET_TEXT)
        features = extractor.extract_from_file(args.raw_data)

        # Save extracted features
        ensure_dir(DATA_PROCESSED_PATH)
        output_path = os.path.join(DATA_PROCESSED_PATH, 'features.npy')
        save_numpy(features, output_path)
        logger.info(f"Saved extracted features to {output_path}")
    else:
        # Try default path
        default_path = os.path.join(DATA_PROCESSED_PATH, 'features.npy')
        if os.path.exists(default_path):
            logger.info(f"Loading features from default path: {default_path}")
            features = load_numpy(default_path)
        else:
            logger.error("No data provided. Use --data or --raw-data")
            return

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Number of samples: {len(features)}")

    feature_dim = features.shape[1]

    # Initialize GAN
    logger.info(f"Initializing GAN: latent_dim={args.latent_dim}, feature_dim={feature_dim}")
    gan = KeystrokeGAN(latent_dim=args.latent_dim, feature_dim=feature_dim)

    # Print model summaries
    logger.info("\nGenerator Summary:")
    gan.generator.summary(print_fn=logger.info)

    logger.info("\nDiscriminator Summary:")
    gan.discriminator.summary(print_fn=logger.info)

    # Train GAN
    logger.info(f"\nTraining GAN: {args.epochs} epochs, batch_size={args.batch_size}")

    # Estimate training time
    if has_gpu:
        estimated_time = args.epochs * 0.3  # ~0.3 seconds per epoch on GPU
        logger.info(f"Estimated training time: ~{estimated_time/60:.1f} minutes (with GPU)")
    else:
        estimated_time = args.epochs * 1.5  # ~1.5 seconds per epoch on CPU
        logger.info(f"Estimated training time: ~{estimated_time/60:.1f} minutes (with CPU)")

    logger.info("This may take a while...")

    history = gan.train(
        X_real=features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=100
    )

    logger.info("Training complete!")

    # Save model
    if args.output:
        model_path = args.output
    else:
        ensure_dir(MODEL_PATH)
        model_path = os.path.join(MODEL_PATH, 'gan_model.h5')

    gan.save_models(model_path)
    logger.info(f"Model saved to {model_path}")

    # Generate test samples
    logger.info("Generating test samples...")
    n_test_samples = 10
    test_samples = gan.generate(n_test_samples)

    test_samples_path = os.path.join(MODEL_PATH, 'gan_test_samples.npy')
    save_numpy(test_samples, test_samples_path)
    logger.info(f"Test samples saved to {test_samples_path}")

    # Log sample statistics
    logger.info(f"Test samples shape: {test_samples.shape}")
    logger.info(f"Test samples mean: {np.mean(test_samples):.4f}")
    logger.info(f"Test samples std: {np.std(test_samples):.4f}")

    # Plot training history
    history_plot_path = os.path.join(MODEL_PATH, 'gan_training_history.png')
    plot_training_history(history, history_plot_path)

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': features.shape,
        'n_samples': len(features),
        'feature_dim': feature_dim,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'final_d_loss': float(history['d_loss'][-1]),
        'final_g_loss': float(history['g_loss'][-1]),
        'final_d_acc': float(history['d_acc'][-1]),
        'model_path': model_path,
        'test_samples_path': test_samples_path,
        'test_samples_stats': {
            'mean': float(np.mean(test_samples)),
            'std': float(np.std(test_samples)),
            'min': float(np.min(test_samples)),
            'max': float(np.max(test_samples))
        }
    }

    report_path = os.path.join(MODEL_PATH, 'gan_training_report.json')
    save_json(report, report_path)
    logger.info(f"Training report saved to {report_path}")

    logger.info("="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)


if __name__ == '__main__':
    main()
