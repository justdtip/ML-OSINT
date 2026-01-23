# ML_OSINT

Multi-source OSINT (Open Source Intelligence) analysis pipeline for conflict prediction using machine learning.

## Overview

ML_OSINT integrates diverse data sources - satellite imagery, conflict event databases, military equipment losses, social media, and narrative sources - to build predictive models for conflict dynamics. The core architecture uses Hierarchical Attention Networks (HAN) with multi-resolution temporal analysis.

### Key Features

- **Multi-source data fusion**: UCDP events, NASA FIRMS fire hotspots, DeepState territorial maps, Sentinel satellite data, equipment/personnel losses
- **Temporal interpolation**: Neural methods for filling gaps in sparse, irregular OSINT data
- **Hierarchical Attention Networks**: Multi-domain attention mechanisms for learning cross-source relationships
- **Multi-resolution analysis**: Daily, weekly, and monthly temporal scales
- **Model interpretability**: Comprehensive probe system for validation and understanding

## Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA or MPS (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ML_OSINT.git
cd ML_OSINT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Data Acquisition

See [data/README.md](data/README.md) for detailed instructions on acquiring each data source.

```bash
# Download VIIRS nightlights data
python scripts/download_viirs_nightlights.py

# Fetch DeepState snapshots
python scripts/fetch_deepstate_live.py
```

### Training Models

```bash
# Train the Hierarchical Attention Network
python -m analysis.train_han --config production

# Train multi-resolution models
python -m analysis.train_multi_resolution --epochs 100

# Run the complete pipeline
python -m analysis.train_full_pipeline
```

### Making Predictions

```bash
# Generate predictions with trained model
python -m analysis.predict

# Predict next tactical state
python -m analysis.predict_next_state
```

### Running Validation Probes

```bash
# Run all probes
python -m analysis.probes.run_probes

# Run specific tier
python -m analysis.probes.run_probes --tier 1
```

## Project Structure

```
ML_OSINT/
├── analysis/                 # Core ML pipeline
│   ├── models/              # Trained model checkpoints
│   ├── checkpoints/         # Training state
│   ├── probes/              # Model validation
│   └── reports/             # Analysis reports
├── config/                   # Centralized configuration
│   ├── paths.py             # Path constants
│   └── logging_config.py    # Logging setup
├── data/                     # All data sources
│   ├── ucdp/                # Conflict events
│   ├── firms/               # Fire hotspots
│   ├── deepstate/           # Territorial control
│   └── [other sources]/
├── scripts/                  # Data download utilities
├── logs/                     # Log files
├── outputs/                  # Generated outputs
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project guidelines and conventions
- [docs/HAN.md](docs/HAN.md) - Hierarchical Attention Network architecture
- [docs/Probe-specs.md](docs/Probe-specs.md) - Model validation probe specifications
- [data/README.md](data/README.md) - Data acquisition guide
- [docs/README.md](docs/README.md) - Documentation index

## Data Sources

| Source | Type | Update Frequency |
|--------|------|------------------|
| UCDP GED | Conflict events | Monthly |
| NASA FIRMS | Fire hotspots | Daily |
| DeepState | Territorial control | Daily |
| Sentinel | Satellite imagery | Weekly |
| War Losses | Equipment/Personnel | Daily |
| ISW | Narrative assessments | Daily |

## Model Architecture

The core model uses a Hierarchical Attention Network with:

1. **Domain encoders**: Separate encoders for each data source
2. **Cross-domain attention**: Learning relationships between sources
3. **Temporal modeling**: Multi-resolution temporal patterns
4. **State transition**: Modeling conflict phase transitions
5. **Uncertainty estimation**: Calibrated prediction confidence

See [docs/HAN.md](docs/HAN.md) for detailed architecture documentation.

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
VOYAGE_API_KEY=your_voyage_key
```

### Training Configuration

Edit `analysis/training_config.py` or use presets:

```python
from analysis.training_config import ExperimentConfig

config = ExperimentConfig.from_preset("production")
```

Available presets: `default`, `fast_debug`, `production`, `high_capacity`

## Contributing

1. Read [CLAUDE.md](CLAUDE.md) for project conventions
2. Use `config/paths.py` for all path references
3. Follow the module hierarchy to avoid circular dependencies
4. Run tests before submitting changes
5. Update documentation for new features

## License

[License information here]

## Acknowledgments

- UCDP for conflict event data
- NASA for FIRMS and VIIRS data
- DeepState for territorial control maps
- ISW for daily assessments
