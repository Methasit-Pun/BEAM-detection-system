#!/usr/bin/env python3
"""
E-Scooter Safety Detection System
Main entry point for the detection system

Usage:
    python detect_e_scooter.py --config config.yaml
    python detect_e_scooter.py --camera csi://0
    python detect_e_scooter.py --model models/trained-model/ssd-mobilenet.onnx
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.detector import EScooterDetector
from src.utils import load_config, setup_logging, create_directories, print_system_info


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='E-Scooter Safety Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python detect_e_scooter.py
  
  # Run with custom config
  python detect_e_scooter.py --config my_config.yaml
  
  # Override camera source
  python detect_e_scooter.py --camera /dev/video0
  
  # Override model path
  python detect_e_scooter.py --model models/my_model.onnx
  
  # Show system info
  python detect_e_scooter.py --info
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--camera',
        type=str,
        help='Override camera source (e.g., csi://0, /dev/video0, 0)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Override model path'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        help='Override labels file path'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display window (headless mode)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Show system info if requested
    if args.info:
        print_system_info()
        return 0
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.camera:
            config['camera']['source'] = args.camera
            print(f"Camera source overridden: {args.camera}")
        
        if args.model:
            config['model']['path'] = args.model
            print(f"Model path overridden: {args.model}")
        
        if args.labels:
            config['model']['labels'] = args.labels
            print(f"Labels path overridden: {args.labels}")
        
        if args.no_display:
            config['display']['show_window'] = False
            print("Display window disabled")
        
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
            config['debug']['verbose'] = True
        
        # Create necessary directories
        create_directories(config)
        
        # Setup logging
        setup_logging(config)
        
        # Initialize and run detector
        print("\n" + "=" * 60)
        print("E-Scooter Safety Detection System")
        print("=" * 60)
        print(f"Camera: {config['camera']['source']}")
        print(f"Model: {config['model']['path']}")
        print(f"Labels: {config['model']['labels']}")
        print("=" * 60)
        print("\nPress 'q' to quit\n")
        
        detector = EScooterDetector(config)
        detector.run()
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure the following files exist:")
        print("  - config.yaml (or specified config file)")
        print("  - Model file (e.g., models/trained-model/ssd-mobilenet.onnx)")
        print("  - Labels file (e.g., models/trained-model/labels.txt)")
        return 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
