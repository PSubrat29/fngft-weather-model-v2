
# FNGFT-AI Weather Model v2 - GitHub Repository Setup Guide

This document provides a comprehensive step-by-step guide for setting up the FNGFT-AI Weather Model v2 as a proper GitHub repository. It covers initialization, configuration, best practices, and deployment instructions.

## Table of Contents

1. [Initial Repository Setup](#initial-repository-setup)
2. [Repository Structure](#repository-structure)
3. [Git Configuration](#git-configuration)
4. [Dependency Management](#dependency-management)
5. [Environment Setup](#environment-setup)
6. [Data Configuration](#data-configuration)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Inference and Forecasting](#inference-and-forecasting)
10. [API Deployment](#api-deployment)
11. [Cloud Deployment](#cloud-deployment)
12. [Testing](#testing)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## Initial Repository Setup

### 1. Initialize Git Repository

```bash
# Navigate to project directory
cd D:\ParidaUser\Claude-Project\fngft_weather_model_v2

# Initialize git repository
git init

# Configure user information (if not already set)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: FNGFT-AI Weather Model v2 prototype"
```

### 2. Create .gitignore File

Create a `.gitignore` file in the root directory with the following content:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.pyo
*.pyd
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
venv/
ENV/
env.bak/
venv.bak/

# PyTorch
*.pth
*.pt
*.ckpt

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
tmp_*
temp_*

# Model artifacts
artifacts/

# Cache
.pytest_cache/
.mypy_cache/
.hypothesis/
.coverage
.coverage.*
cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Data files (large datasets should use Git LFS or be excluded)
*.nc
*.nc4
*.grib
*.grb
*.grib2
*.grb2
*.zarr
```

### Set Up Remote Repository on GitHub

1. Create a new repository on GitHub (e.g., `fngft-weather-model-v2`)
2. Add the remote origin:
    
    ```bash
    git remote add origin https://github.com/your-username/fngft-weather-model-v2.git
    ```
    
3. Push to GitHub:
    
    ```bash
    git branch -M main
    git push -u origin main
    ```

---

## Repository Structure

The FNGFT-AI Weather Model v2 follows this structure:

```
fngft_weather_model_v2/
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Containerization configuration
├── GOVERNANCE.md              # Research governance guidelines
├── REAL_DATA_WORKFLOW.md      # Real-data workflow description
├── CLOUD_DEPLOYMENT.md        # Cloud deployment design
├── README.md                  # Project overview and usage instructions
├── REQUIREMENTS.md            # Dependency requirements (alternative format)
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Python package dependencies
├── pytest.ini                 # Pytest configuration
│
├── fngft/                     # Main Python package
│   ├── __init__.py            # Package initializer
│   ├── __init__.md            # Package documentation
│   ├── api.py                 # FastAPI inference service
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration management
│   ├── evaluate.py            # Model evaluation utilities
│   ├── io.py                  # Data loading and inspection
│   ├── losses.py              # Loss functions
│   ├── model.py               # Neural network architecture
│   ├── operators.py           # Fractional memory operators
│   ├── physics.py             # Physics component
│   ├── preprocess.py          # Data preprocessing
│   ├── realtime.py            # Real-time forecasting
│   └── schema.py              # Data validation schemas
│
├── configs/                   # Configuration files
│   └── real_data.yaml         # Example data configuration
│
├── tests/                     # Test suite
│   ├── __pycache__/           # Python cache
│   └── test_pipeline.py       # Integration tests
│
├── artifacts/                 # Model checkpoints and outputs (gitignored)
│   └── fngft_real.pt          # Example model checkpoint
│
├── tmp_verify_pipeline.py     # Temporary verification script
│
└── .venv/                     # Virtual environment (gitignored)
```

---

## Git Configuration

### 1. Set Up Git Hooks (Optional)

For maintaining code quality, consider setting up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_args: [--ignore-missing-imports]
EOF

# Install hooks
pre-commit install
```

### 2. Configure Git Attributes (For Large Files)

If you plan to store large model files or datasets in Git (not recommended for production), set up Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"
git lfs track "*.nc"
git lfs track "*.zarr"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for large files"
```

---

## Dependency Management

### 1. Using pyproject.toml (Recommended)

The project uses modern Python packaging with `pyproject.toml`:

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### 2. Using requirements.txt

For traditional dependency management:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Update requirements.txt after changes
pip freeze > requirements.txt
```

### 3. Development Dependencies

Add these to your development environment:
```bash
pip install pytest black flake8 mypy pre-commit
```

---

## Environment Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Verify Installation

```bash
# Check if package is installed correctly
python -c "import fngft; print(fngft.__version__)"
```

### 4. Jupyter Notebook Support (Optional)

```bash
pip install notebook jupyterlab
```

---

## Data Configuration

### 1. Prepare Your Data

The model requires gridded weather data with the following specifications:
- Dimensions: `time`, `lat`, `lon`
- Variables: `u` (zonal wind), `v` (meridional wind), `theta` (potential temperature), `q` (specific humidity)
- Supported formats: NetCDF (.nc, .nc4), Zarr (.zarr), GRIB (.grib, .grb, .grib2, .grb2)

### 2. Configure Data Source

Edit `configs/real_data.yaml` to match your data:

```yaml
data:
  source: /path/to/your/weather/data.nc  # Update with your data path
  format: auto                          # or specify: netcdf, zarr, grib
  engine: null                          # Specify engine if needed (e.g., scipy for NetCDF)
  time_dim: time
  lat_dim: lat
  lon_dim: lon
  variables:
    u: your_u_variable_name             # Map to your u variable
    v: your_v_variable_name             # Map to your v variable
    theta: your_temperature_variable    # Map to your temperature variable
    q: your_humidity_variable           # Map to your humidity variable
  level_dim: null                       # Set if data has vertical levels
  level_value: null                     # Value to select if level_dim is set
  time_step_hours: null                 # Set if known, otherwise inferred
  train_start: "2024-01-01"             # Training period start
  train_end: "2025-12-31T23:59:59"      # Training period end
  val_start: "2026-01-01"               # Validation period start
  val_end: "2026-06-30T23:59:59"        # Validation period end
  test_start: "2026-07-01"              # Test period start
  test_end: "2026-09-30T23:59:59"       # Test period end
  longitude_periodic: true              # Set to false for non-global domains

model:
  # Model hyperparameters (adjust based on your data and resources)
  in_channels: 4
  hidden: 48
  memory_dim: 64
  memory_heads: 4
  memory_layers: 2
  history: 8
  # ... other parameters as needed

training:
  batch_size: 4
  epochs: 5
  learning_rate: 0.0002
  # ... other training parameters
```

### 3. Inspect Your Dataset

Before training, validate your dataset configuration:
```bash
python -m fngft.cli inspect --config configs/real_data.yaml
```

This will output a dataset profile including:
- Time range and step
- Latitude/longitude bounds and resolution
- Variable mapping validation
- Grid regularity checks

---

## Model Training

### 1. Start Training

```bash
python -m fngft.cli train --config configs/real_data.yaml
```

Training process:
1. Loads and validates the dataset
2. Fits normalization statistics on training data only
3. Creates temporal windows for training/validation
4. Trains the hybrid FNGFT-AI model
5. Saves the best checkpoint to `artifacts/fngft_real.pt`
6. Includes model/data/grid metadata in checkpoint

### 2. Monitor Training

Training output shows:
```
epoch=1 train_loss=0.123456 val_loss=0.134567
epoch=2 train_loss=0.112345 val_loss=0.123456
...
```

### 3. Training Tips

- Start with small datasets for testing
- Adjust `batch_size` based on GPU memory
- Modify `epochs` based on convergence
- Use learning rate scheduling for longer training
- Monitor validation loss to prevent overfitting

---

## Model Evaluation

### 1. Evaluate on Test Set

```bash
python -m fngft.cli evaluate \
  --config configs/real_data.yaml \
  --checkpoint artifacts/fngft_real.pt \
  --split test
```

### 2. Evaluation Metrics

The evaluation script returns:
- `rmse_standardized`: Root Mean Square Error (standardized space)
- `mae_standardized`: Mean Absolute Error (standardized space)
- `anomaly_correlation`: Anomaly correlation coefficient
- `alpha_mean`, `beta_mean`, `kappa_mean`: Average order parameters

### 3. Evaluate on Different Splits

```bash
# Evaluate on training set
python -m fngft.cli evaluate --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --split train

# Evaluate on validation set
python -m fngft.cli evaluate --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --split val
```

### 4. Custom Evaluation Output

Save results to a file:
```bash
python -m fngft.cli evaluate \
  --config configs/real_data.yaml \
  --checkpoint artifacts/fngft_real.pt \
  --split test \
  --output evaluation_results.json
```

---

## Inference and Forecasting

### 1. Latest File Forecast

For real-time forecasting with the newest data file:
```bash
python -m fngft.cli forecast-latest \
  --config configs/real_data.yaml \
  --checkpoint artifacts/fngft_real.pt \
  --steps 6 \
  --output artifacts/latest_forecast.npz
```

This:
1. Finds the newest file in the data source directory
2. Validates time step compatibility
3. Creates history window from most recent observations
4. Generates forecast for specified steps
5. Saves standardized and physical-space forecasts

### 2. Forecast Output Format

The `.npz` file contains:
- `forecast`: Physical space forecast [steps, 4, lat, lon]
- `time`: Forecast timestamps
- `lat`: Latitude coordinates
- `lon`: Longitude coordinates
- `alpha`, `beta`, `kappa`: Order parameter maps

### 3. One-Step Forecast

```bash
python -m fngft.cli forecast-latest \
  --config configs/real_data.yaml \
  --checkpoint artifacts/fngft_real.pt \
  --steps 1 \
  --output artifacts/one_step_forecast.npz
```

---

## API Deployment

### 1. Start the HTTP API

```bash
uvicorn fngft.api:app --host 0.0.0.0 --port 8080
```

Or with explicit model path:
```bash
MODEL_PATH=artifacts/fngft_real.pt uvicorn fngft.api:app --host 0.0.0.0 --port 8080
```

### 2. API Endpoints

- `GET /health`: Health check
- `GET /model-info`: Model configuration information
- `POST /forecast`: Generate forecast

### 3. Forecast Request Example

```json
{
  "history": [[[...]]],  // Standardized state: [time=8, 4, lat_points, lon_points]
  "lat": [....],         // Latitude coordinates
  "lon": [....],         // Longitude coordinates
  "steps": 6             // Forecast steps (1-24)
}
```

### 4. Forecast Response Example

```json
{
  "forecast_standardized": [[[...]]],  // [steps, 4, lat, lon]
  "alpha_mean": 0.45,
  "beta_mean": 0.65,
  "kappa_mean": 0.12,
  "alpha_map": [[...]],    // [lat, lon]
  "beta_map": [[...]],     // [lat, lon]
  "kappa_map": [[...]]     // [lat, lon]
}
```

### 5. Docker Deployment

Build and run the container:
```bash
# Build Docker image
docker build -t fngft-weather-model .

# Run container
docker run -p 8080:8080 fngft-weather-model
```

Or with volume mounting for checkpoints:
```bash
docker run -p 8080:8080 -v $(pwd)/artifacts:/app/artifacts fngft-weather-model
```

---

## Cloud Deployment

Following the architecture in `CLOUD_DEPLOYMENT.md`:

### 1. Data Plane

- Store raw data in object storage (S3, GCS, Azure Blob)
- Ingestion jobs validate and standardize data
- Store standardized datasets for training

### 2. Training Plane

- GPU-enabled training jobs
- Produce immutable model artifacts
- Register checkpoints with metadata in model registry

### 3. Inference Plane

- Containerized inference service
- Horizontal scaling via load balancer
- Serve approved model versions only

### 4. Governance Plane

- Track data provenance (source, version, retrieval time)
- Record model provenance (code revision, config, seed, hardware)
- Store validation metrics and release status

### 5. Example Workflow

1. New data arrives in object storage
2. Scheduler triggers ingestion/validation job
3. Validated data becomes available for training
4. Training job creates new checkpoint
5. Checkpoint undergoes validation gates (see GOVERNANCE.md)
6. Approved checkpoint is deployed to inference service
7. Forecast API serves predictions from approved model

---

## Testing

### 1. Run Test Suite

```bash
pytest -q
```

### 2. Verbose Testing

```bash
pytest -v
```

### 3. Test Coverage (Optional)

```bash
pip install pytest-cov
pytest --cov=fngft --cov-report=html
```

### 4. What Tests Cover

The test suite includes:
- Dataset creation and validation
- Configuration loading
- Data preprocessing pipeline
- Model architecture verification
- Training/evaluation workflow
- Forecasting functionality

---

## Best Practices

### 1. Data Management

- Always inspect data before training
- Use time-normalized splits (train/val/test)
- Store only training statistics in normalizer
- Keep raw data immutable
- Version your datasets

### 2. Model Development

- Follow the real-data workflow (REAL_DATA_WORKFLOW.md)
- Adhere to governance guidelines (GOVERNANCE.md)
- Validate checkpoints before promotion
- Monitor alpha/beta/kappa stability
- Check spectral behavior and gradient consistency

### 3. Reproducibility

- Set random seeds (42 by default)
- Archive exact dependency versions
- Store complete configuration with checkpoints
- Document hardware and environment
- Use deterministic algorithms when possible

### 4. Code Quality

- Follow existing code style
- Write documentation for new modules
- Add type hints where beneficial
- Create unit tests for new functionality
- Use pre-commit hooks for formatting

### 5. Security

- Never accept arbitrary file paths in API
- Validate all inputs
- Use environment variables for secrets
- Keep dependencies updated
- Follow principle of least privilege

---

## Troubleshooting

### 1. Common Issues and Solutions

#### Dataset Loading Errors

- **Error**: `Variable 'name' not found in dataset`
  - **Solution**: Check variable mapping in `configs/real_data.yaml`
- **Error**: `Required dimension 'dim' not found`
  - **Solution**: Verify dimension names in your dataset
- **Error**: `Time coordinate must be strictly increasing`
  - **Solution**: Sort your time dimension or fix data source

#### Training Issues

- **Error**: `CUDA out of memory`
  - **Solution**: Reduce `batch_size` or `hidden` size
- **Error**: `NaN losses`
  - **Solution**: Check data for invalid values, reduce learning rate
- **Error**: `Validation loss not decreasing`
  - **Solution**: Check for data leakage, verify normalization

#### API Issues

- **Error**: `Model is not loaded`
  - **Solution**: Ensure `MODEL_PATH` points to valid checkpoint
- **Error**: `history must have shape [time,4,lat,lon]`
  - **Solution**: Verify input dimensions match model expectations
- **Error**: `lat/lon lengths do not match history grid`
  - **Solution**: Ensure coordinate arrays match spatial dimensions

#### Docker Issues

- **Error**: `Model file not found`
  - **Solution**: Check volume mounting or build process includes artifacts
- **Error**: `Address already in use`
  - **Solution**: Stop existing containers or change port

### 2. Getting Help

- Check the README.md for basic usage
- Refer to specific module documentation (.md files next to .py files)
- Review governance documents for scientific guidelines
- Consult the real-data workflow and cloud deployment designs
- Search existing issues or create new ones on GitHub

---

## Conclusion

This guide provides a comprehensive approach to setting up the FNGFT-AI Weather Model v2 as a proper GitHub repository. By following these steps, you'll ensure:

1. Proper version control and collaboration
2. Reproducible research and development
3. Clean separation of concerns
4. Easy deployment to various environments
5. Adherence to scientific best practices
6. Maintainable and extensible codebase

Remember that this is a research prototype, and any operational use should follow the governance guidelines and undergo proper validation. The model is designed for experimentation with real gridded weather datasets and serves as a foundation for further research in fractional calculus-based weather forecasting.

Happy modeling!