#!/bin/bash

# Exit on error
set -e

echo "Setting up EOPF 101 development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python version must be 3.9 or higher"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
pip install -e .

# Install Jupyter kernel
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=eopf101

# Check if Quarto is installed
if ! command -v quarto &> /dev/null; then
    echo "Warning: Quarto is not installed. Please install it from https://quarto.org/docs/get-started/"
fi

echo "Verifying installation..."
python -c "import zarr; import pystac; import rioxarray; import stackstac; print('Environment setup successful!')"

echo "Setup complete! To start working:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Start Jupyter: jupyter notebook"
echo "3. Preview documentation: quarto preview"
