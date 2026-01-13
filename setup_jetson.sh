#!/bin/bash
# Setup script for NVIDIA Jetson Nano
# E-Scooter Safety Detection System

set -e

echo "=========================================="
echo "E-Scooter Detection System - Jetson Setup"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This doesn't appear to be a Jetson device"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo ""
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "[2/8] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libportaudio2 \
    portaudio19-dev \
    libasound2-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev

# Create virtual environment
echo ""
echo "[3/8] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/8] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson
echo ""
echo "[5/8] Installing PyTorch for Jetson..."
# Check JetPack version and install appropriate PyTorch wheel
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision
pip install torchvision

# Install Python dependencies
echo ""
echo "[6/8] Installing Python dependencies..."
pip install -r requirements.txt

# Setup jetson-inference
echo ""
echo "[7/8] Setting up jetson-inference..."
if [ ! -d "jetson-inference" ]; then
    echo "Cloning jetson-inference repository..."
    git clone --recursive https://github.com/dusty-nv/jetson-inference.git
    cd jetson-inference
    mkdir -p build
    cd build
    cmake ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd ../..
else
    echo "jetson-inference already exists, skipping..."
fi

# Create directory structure
echo ""
echo "[8/8] Creating directory structure..."
mkdir -p models/trained-model
mkdir -p data/{train,val,test}
mkdir -p audio
mkdir -p logs
mkdir -p results

# Create placeholder files
echo "# Place your trained model here: ssd-mobilenet.onnx" > models/trained-model/README.md
echo "# Place your alert sound here: violation_alert.wav" > audio/README.md
echo "# Training data goes here" > data/README.md

# Download sample alert sound (if available)
echo ""
echo "Note: You need to add your alert sound file to audio/violation_alert.wav"

# Set permissions
echo ""
echo "Setting file permissions..."
chmod +x detect_e_scooter.py
chmod +x scripts/*.sh 2>/dev/null || true

# Test imports
echo ""
echo "Testing Python imports..."
python3 << EOF
import sys
print("Python version:", sys.version)

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError:
    print("✗ OpenCV not found")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except ImportError:
    print("✗ PyTorch not found")

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except ImportError:
    print("✗ NumPy not found")

try:
    import onnxruntime
    print("✓ ONNX Runtime:", onnxruntime.__version__)
except ImportError:
    print("✗ ONNX Runtime not found")
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your trained model in: models/trained-model/"
echo "2. Place your alert sound in: audio/violation_alert.wav"
echo "3. Update config.yaml with your settings"
echo "4. Run: python3 detect_e_scooter.py"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the detection system:"
echo "  python3 detect_e_scooter.py --config config.yaml"
echo ""
