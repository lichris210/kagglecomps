#!/bin/bash
# Quick setup for the Kaggle S6E4 project
# Usage: bash setup.sh

set -e

echo "=== Setting up Kaggle S6E4 environment ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages
echo "Installing dependencies..."
pip install -r requirements.txt

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name kaggle-s6e4 --display-name "Kaggle S6E4"

# Download data if kaggle CLI is configured
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Downloading competition data..."
    kaggle competitions download -c playground-series-s6e4 -p data/
    cd data && unzip -o playground-series-s6e4.zip && rm -f playground-series-s6e4.zip && cd ..
    echo "Data downloaded to data/"
else
    echo ""
    echo "NOTE: Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "To download data automatically:"
    echo "  1. Go to kaggle.com -> Settings -> API -> Create New Token"
    echo "  2. Move kaggle.json to ~/.kaggle/kaggle.json"
    echo "  3. chmod 600 ~/.kaggle/kaggle.json"
    echo "  4. Re-run this script or run:"
    echo "     kaggle competitions download -c playground-series-s6e4 -p data/"
    echo ""
    echo "Or download manually from the competition page and put CSVs in data/"
fi

echo ""
echo "=== Setup complete! ==="
echo "To get started:"
echo "  source venv/bin/activate"
echo "  jupyter notebook"
