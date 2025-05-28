# Workspace Setup Guide

This guide will help you set up your development environment to run the EOPF 101 notebooks and interact with Sentinel data in Zarr format.

## Prerequisites

- Python 3.9 or higher
- Git
- pip (Python package installer)

## Setting Up the Environment

### Automated Setup (Linux/macOS)

1. Clone the repository:
```bash
git clone https://github.com/eopf-toolkit/eopf-101
cd eopf-101
```

2. Run the setup script:
```bash
./setup.sh
```

This script will:
- Check Python version requirements
- Create and activate a virtual environment
- Install all dependencies
- Set up Jupyter kernel
- Verify the installation

### Manual Setup (Windows or Alternative)

1. Clone the repository:
```bash
git clone https://github.com/eopf-toolkit/eopf-101
cd eopf-101
```

2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install the required dependencies:
```bash
python -m pip install --upgrade pip
pip install -e .
```

## Project Configuration

Create a `pyproject.toml` file in your project root with the following content:

```toml
[project]
name = "eopf-101"
version = "0.1.0"
description = "EOPF 101 - A toolkit for working with Sentinel data in Zarr format"
requires-python = ">=3.9"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "-ra -q"
testpaths = ["tests"]

[project.dependencies]
jupyter = "^1.0.0"
zarr = "^2.14.2"
pystac = "^1.7.3"
numpy = "^1.24.3"
pandas = "^2.0.2"
matplotlib = "^3.7.1"
rioxarray = "^0.14.1"
stackstac = "^0.4.4"
ipykernel = "^6.23.1"
```

## Required Dependencies

Create a `requirements.txt` file with the following content:

```txt
jupyter>=1.0.0
zarr>=2.14.2
pystac>=1.7.3
numpy>=1.24.3
pandas>=2.0.2
matplotlib>=3.7.1
rioxarray>=0.14.1
stackstac>=0.4.4
ipykernel>=6.23.1
quarto>=0.1.0
```

## Setting up Jupyter

1. After installing dependencies, register the Python kernel for Jupyter:
```bash
python -m ipykernel install --user --name=eopf101
```

2. Start Jupyter:
```bash
jupyter notebook
```

## Quarto Setup

Since this project uses Quarto for documentation:

1. Install Quarto from: https://quarto.org/docs/get-started/
2. Preview the documentation:
```bash
quarto preview
```

## Verification

To verify your setup:

1. Ensure your virtual environment is activated
2. Run the following Python commands:

```python
import zarr
import pystac
import rioxarray
import stackstac
print("Environment setup successful!")
```

## Troubleshooting

If you encounter issues:

1. Ensure your Python version is correct:
```bash
python --version
```

2. Verify virtual environment activation:
```bash
# You should see (.venv) at the start of your prompt
```

3. Check installed packages:
```bash
pip list
```

## Getting Help

If you need assistance:
- Open an issue on our GitHub repository
- Check our documentation at https://github.com/eopf-toolkit/eopf-101
- Refer to the README.md for project overview
