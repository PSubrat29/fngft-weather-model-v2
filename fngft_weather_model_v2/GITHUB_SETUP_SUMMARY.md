# FNGFT-AI Weather Model v2 - GitHub Setup Summary

## Quick Start Guide

### 1. Initialize Repository
```bash
cd D:\ParidaUser\Claude-Project\fngft_weather_model_v2
git init
git add .
git commit -m "Initial commit: FNGFT-AI Weather Model v2"
```

### 2. Create .gitignore
Create `.gitignore` with:
```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
.env/
.venv/
env/
venv/
ENV/
env.bak/
venv.bak/

# PyTorch
*.pth
*.pt
*.ckpt

# Jupyter
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

# Temporary
tmp_*
temp_*

# Artifacts
artifacts/

# Cache
.pytest_cache/
.mypy_cache/
.hypothesis/
.coverage
.cache
```

### 3. Set Up GitHub Remote
```bash
git remote add origin https://github.com/your-username/fngft-weather-model-v2.git
git branch -M main
git push -u origin main
```

### 4. Install Dependencies
```bash
# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### 5. Configure Your Data
Edit `configs/real_data.yaml`:
- Set `source` to your weather data path
- Map variables: `u`, `v`, `theta`, `q` to your dataset variable names
- Set training/validation/test date ranges

### 6. Validate Data
```bash
python -m fngft.cli inspect --config configs/real_data.yaml
```

### 7. Train Model
```bash
python -m fngft.cli train --config configs/real_data.yaml
```

### 8. Evaluate Model
```bash
python -m fngft.cli evaluate --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --split test
```

### 9. Generate Forecast
```bash
python -m fngft.cli forecast-latest --config configs/real_data.yaml --checkpoint artifacts/fngft_real.pt --steps 6
```

### 10. Run API
```bash
uvicorn fngft.api:app --host 0.0.0.0 --port 8080
```

## Key Files
- `README.md`: Project overview
- `pyproject.toml` / `requirements.txt`: Dependencies
- `fngft/`: Main source code
- `configs/`: Configuration templates
- `tests/`: Test suite
- `GOVERNANCE.md`: Research guidelines
- `REAL_DATA_WORKFLOW.md`: Workflow description
- `CLOUD_DEPLOYMENT.md`: Deployment architecture

## Best Practices
1. Always inspect data before training
2. Use time-normalized train/val/test splits
3. Store only training statistics in normalizer
4. Follow governance guidelines for model promotion
5. Keep raw data immutable
6. Set random seeds for reproducibility
7. Use virtual environments
8. Write documentation for new code
9. Run tests before committing
10. Never accept arbitrary file paths in API

## Troubleshooting
- Dataset errors: Check variable mapping and dimension names
- Training errors: Reduce batch size, check for NaN values
- API errors: Verify MODEL_PATH and input dimensions
- Docker errors: Check volume mounting and port availability

Happy modeling with FNGFT-AI!