#!/usr/bin/env python3
"""
Create sample keystroke data for testing
Works on Windows, Linux, and Mac
"""

import numpy as np
import json
import os
from datetime import datetime

def create_sample_data(output_path='data/raw/sample_data.json', n_samples=100):
    """
    Generate sample keystroke data

    Args:
        output_path: Path to save data
        n_samples: Number of samples to generate
    """
    print("="*60)
    print("CREATING SAMPLE KEYSTROKE DATA")
    print("="*60)

    target_text = 'security2025'
    samples = []

    print(f"\nGenerating {n_samples} samples...")
    print(f"Target text: '{target_text}'")

    for i in range(n_samples):
        keypresses = []
        time = 0

        for j, char in enumerate(target_text):
            # Random dwell time (50-150ms)
            dwell = np.random.uniform(0.05, 0.15)

            keypresses.append({
                'char': char,
                'position': j,
                'keydown_time': time,
                'keyup_time': time + dwell
            })

            # Random flight time (100-200ms)
            time += dwell + np.random.uniform(0.1, 0.2)

        samples.append({
            'sample_id': i + 1,
            'timestamp': datetime.now().isoformat(),
            'keypresses': keypresses
        })

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")

    # Create data structure
    data = {
        'metadata': {
            'user_id': 'instructor',
            'target_text': target_text,
            'collection_date': datetime.now().isoformat(),
            'num_samples': len(samples)
        },
        'samples': samples
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Sample data created successfully!")
    print(f"✓ Location: {output_path}")
    print(f"✓ Samples: {len(samples)}")
    print(f"✓ Characters per sample: {len(target_text)}")
    print("\nYou can now train models with this data:")
    print(f"  python -m src.training.train_auth --raw-data {output_path}")
    print(f"  python -m src.training.train_gan --raw-data {output_path}")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create sample keystroke data')
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/sample_data.json',
        help='Output file path'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )

    args = parser.parse_args()

    create_sample_data(args.output, args.samples)
