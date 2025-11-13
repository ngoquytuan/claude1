#!/usr/bin/env python3
"""
Main entry point for Keystroke GAN Demo
Run this script to start the demo application
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.demo.demo_ui import DemoUI
from src.utils.logger import setup_logger

logger = setup_logger('main')


def main():
    """Main function"""
    logger.info("="*60)
    logger.info("KEYSTROKE AUTHENTICATION + GAN BYPASS DEMO")
    logger.info("="*60)

    try:
        # Initialize and run demo UI
        logger.info("Initializing demo UI...")
        demo = DemoUI()

        logger.info("Starting demo application...")
        demo.run()

        logger.info("Demo application closed")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
