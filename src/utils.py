"""
Utility Functions
Helper functions for the detection system
"""

import cv2
import logging
import os
import yaml
from datetime import datetime


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary with logging settings
    """
    log_config = config.get('logging', {})
    
    if not log_config.get('enabled', True):
        return
    
    # Create logs directory if needed
    log_dir = log_config.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = os.path.join(log_dir, log_config.get('log_file', 'detection.log'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging initialized")


def draw_detections(frame, detections, show_confidence=True, thickness=2):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        show_confidence: Show confidence scores
        thickness: Line thickness
    
    Returns:
        Annotated frame
    """
    for det in detections:
        bbox = det.get('bbox', [])
        class_name = det.get('class_name', 'unknown')
        confidence = det.get('confidence', 0.0)
        
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on class
        if 'scooter' in class_name.lower() or 'beam' in class_name.lower():
            color = (0, 255, 0)  # Green for scooters
        elif 'person' in class_name.lower():
            color = (255, 0, 0)  # Blue for people
        else:
            color = (128, 128, 128)  # Gray for others
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = class_name
        if show_confidence:
            label += f" {confidence:.2f}"
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return frame


def save_violation_log(violation_data, log_file='logs/violations.csv'):
    """
    Save violation data to CSV log
    
    Args:
        violation_data: Violation information
        log_file: Path to log file
    """
    import csv
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Check if file exists to write header
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='') as f:
        fieldnames = ['timestamp', 'rider_count', 'scooter_confidence', 'scooter_bbox']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        timestamp = datetime.fromtimestamp(violation_data.get('timestamp', 0))
        scooter = violation_data.get('scooter', {})
        
        writer.writerow({
            'timestamp': timestamp.isoformat(),
            'rider_count': violation_data.get('rider_count', 0),
            'scooter_confidence': scooter.get('confidence', 0.0),
            'scooter_bbox': str(scooter.get('bbox', []))
        })


def create_directories(config):
    """
    Create necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config.get('logging', {}).get('log_dir', 'logs'),
        'models',
        'data',
        'audio',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def check_dependencies():
    """
    Check if required dependencies are installed
    
    Returns:
        dict: Availability of each dependency
    """
    dependencies = {}
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        dependencies['opencv'] = False
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import yaml
        dependencies['pyyaml'] = True
    except ImportError:
        dependencies['pyyaml'] = False
    
    try:
        import onnxruntime
        dependencies['onnxruntime'] = True
    except ImportError:
        dependencies['onnxruntime'] = False
    
    try:
        import pyaudio
        dependencies['pyaudio'] = True
    except ImportError:
        dependencies['pyaudio'] = False
    
    return dependencies


def print_system_info():
    """Print system and dependency information"""
    import sys
    import platform
    
    print("=" * 60)
    print("E-Scooter Detection System - System Information")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    print("\nDependencies:")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name}: {status}")
    
    print("=" * 60)
