"""
Training script for authentication model
"""

import argparse
import numpy as np
import os
from datetime import datetime

from config.settings import DATA_PROCESSED_PATH, MODEL_PATH, TARGET_TEXT
from src.data_collection.feature_extractor import FeatureExtractor
from src.models.authentication import AuthenticationModel
from src.utils.helpers import load_numpy, save_json, ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger('train_auth')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train authentication model')
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
        '--threshold',
        type=float,
        default=0.85,
        help='Authentication threshold (default: 0.85)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for model (default: data/models/auth_model.pkl)'
    )

    args = parser.parse_args()

    logger.info("="*50)
    logger.info("AUTHENTICATION MODEL TRAINING")
    logger.info("="*50)

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
        np.save(output_path, features)
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

    # Split data (80% train, 20% validation)
    n_samples = len(features)
    n_train = int(0.8 * n_samples)

    # Shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train = features[train_indices]
    X_val = features[val_indices]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Train model
    logger.info(f"Training authentication model with threshold={args.threshold}")
    auth_model = AuthenticationModel(threshold=args.threshold)

    train_metrics = auth_model.train(X_train)

    logger.info("Training complete!")
    logger.info(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
    logger.info(f"Decision threshold: {train_metrics['decision_threshold']:.4f}")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_scores = auth_model.predict(X_val)
    val_accuracy = np.mean(val_scores >= args.threshold)

    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    logger.info(f"Validation score - Mean: {np.mean(val_scores):.4f}, Std: {np.std(val_scores):.4f}")

    # Save model
    if args.output:
        model_path = args.output
    else:
        ensure_dir(MODEL_PATH)
        model_path = os.path.join(MODEL_PATH, 'auth_model.pkl')

    auth_model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': features.shape,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'threshold': args.threshold,
        'train_metrics': train_metrics,
        'validation_metrics': {
            'accuracy': float(val_accuracy),
            'score_mean': float(np.mean(val_scores)),
            'score_std': float(np.std(val_scores)),
            'score_min': float(np.min(val_scores)),
            'score_max': float(np.max(val_scores))
        },
        'model_path': model_path
    }

    report_path = os.path.join(MODEL_PATH, 'auth_training_report.json')
    save_json(report, report_path)
    logger.info(f"Training report saved to {report_path}")

    logger.info("="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)


if __name__ == '__main__':
    main()
