# ML_OSINT Project Guidelines

## Project Overview

ML_OSINT is a machine learning pipeline for analyzing and predicting conflict dynamics using multi-source OSINT (Open Source Intelligence) data. The project integrates satellite imagery, conflict event databases, military equipment losses, and narrative sources to build predictive models using Hierarchical Attention Networks (HAN) and multi-resolution temporal analysis.

### Core Capabilities
- **Multi-source data integration**: UCDP, FIRMS, DeepState, Sentinel, ISW assessments, Telegram channels
- **Temporal interpolation**: Filling gaps in sparse, irregular OSINT data
- **Hierarchical Attention Networks**: Multi-domain attention for conflict prediction
- **Multi-resolution analysis**: Daily, weekly, monthly temporal scales
- **Model interpretability**: Comprehensive probe system for validation

---

## Project Structure

```
ML_OSINT/
├── analysis/                    # Core ML pipeline (primary working directory)
│   ├── models/                  # Trained model checkpoints (.pt files)
│   │   ├── interpolation/       # Domain-specific interpolation models
│   │   ├── multi_resolution/    # Multi-resolution HAN checkpoints
│   │   └── pipeline/            # Pipeline stage checkpoints
│   ├── checkpoints/             # Training state checkpoints
│   │   └── multi_resolution/    # Multi-resolution training states
│   ├── probes/                  # Model validation probes
│   │   └── outputs/             # Probe results (figures, CSVs)
│   └── reports/                 # Report generation scripts
├── config/                      # Centralized configuration
│   ├── paths.py                 # Path constants (use this, not hardcoded paths)
│   └── logging_config.py        # Logging setup
├── data/                        # All data sources
│   ├── ucdp/                    # UCDP conflict events
│   ├── firms/                   # NASA FIRMS fire hotspots
│   ├── deepstate/               # DeepState territorial control
│   ├── sentinel/                # Copernicus Sentinel metadata
│   ├── war-losses-data/         # Equipment/personnel losses
│   ├── nasa/viirs_nightlights/  # VIIRS brightness data
│   ├── hdx/ukraine/             # Humanitarian Data Exchange
│   ├── iom/                     # IOM displacement tracking
│   ├── wayback_archives/        # ISW assessments, Google Mobility
│   ├── timelines/               # War timeline annotations + embeddings
│   ├── telegram/                # Telegram channel data + embeddings
│   └── [other sources]/
├── scripts/                     # Data download and utility scripts
├── logs/                        # Centralized log files
├── outputs/                     # Unified output directory
│   ├── figures/                 # All visualizations
│   ├── reports/                 # Generated reports
│   ├── results/                 # CSV/JSON results
│   └── analysis/                # Analysis script outputs
│       ├── cross_source/        # Cross-source analysis
│       ├── delta_model/         # Delta model analysis
│       ├── embeddings/          # Embedding validation
│       ├── features/            # Feature analysis
│       ├── interpretability/    # Interpretability analysis
│       ├── model_assessment/    # Model assessment
│       ├── model_comparison/    # Model comparison
│       ├── network/             # Network analysis
│       └── temporal_prediction/ # Temporal prediction results
├── tests/                       # Unit and integration tests
├── docs/                        # Project documentation
├── telegram_scraper/            # Telegram data collection
├── deepstate-map-data/          # DeepState GeoJSON (git submodule)
├── sen2like/                    # ESA Sentinel harmonization (git submodule)
├── tak-feeder-deepstate/        # TAK server integration (git submodule)
└── omitted_variable_analysis/   # Statistical analysis module
```

---

## Path Configuration

### CRITICAL: Never Hardcode Absolute Paths

All paths MUST be defined relative to the project root or use the central path configuration.

**Use this pattern:**
```python
from config.paths import PROJECT_ROOT, DATA_DIR, MODEL_DIR, OUTPUT_DIR

# Good - uses centralized paths
data_path = DATA_DIR / "ucdp" / "ged_events.csv"
model_path = MODEL_DIR / "han_best.pt"

# Bad - hardcoded absolute path (DO NOT DO THIS)
# data_path = "/Users/daniel.tipton/ML_OSINT/data/ucdp/ged_events.csv"
```

### Standard Path Constants (from config/paths.py)
```python
PROJECT_ROOT = Path(__file__).parent.parent  # /ML_OSINT
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
MODEL_DIR = ANALYSIS_DIR / "models"
INTERP_MODEL_DIR = MODEL_DIR / "interpolation"
CHECKPOINT_DIR = ANALYSIS_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
```

---

## Module Architecture

### Dependency Hierarchy (respect this order)

```
TIER 1: Foundation (no local imports)
├── training_utils.py
├── missing_data_imputation.py
├── feature_selection.py
└── training_targets.py

TIER 2: Core Infrastructure (high reuse)
├── interpolation_data_loaders.py    # 12 dependents
├── joint_interpolation_models.py    # 11 dependents
├── conflict_data_loader.py          # 7 dependents
└── hierarchical_attention_network.py # 7 dependents

TIER 3: Specialized Models
├── unified_interpolation.py         # 7 dependents
├── unified_interpolation_delta.py
├── unified_interpolation_hybrid.py
├── multi_resolution_han.py
└── multi_resolution_data.py

TIER 4: Training Scripts
├── train_han.py
├── train_multi_resolution.py
├── train_full_pipeline.py
└── temporal_prediction.py

TIER 5: Analysis & Visualization (entry points)
├── predict.py / predict_next_state.py
├── visualize_*.py
├── comprehensive_model_report.py
└── model_comparison.py
```

### Key Entry Points
- `train_han.py` - Train Hierarchical Attention Network
- `train_multi_resolution.py` - Train multi-resolution models
- `train_full_pipeline.py` - Run complete training pipeline
- `predict.py` - Generate predictions
- `analysis/probes/run_probes.py` - Run model validation probes

---

## Data Sources

### Primary Sources (actively used in training)
| Source | Location | Format | Purpose |
|--------|----------|--------|---------|
| UCDP | `data/ucdp/ged_events.csv` | CSV | Conflict events |
| FIRMS | `data/firms/DL_FIRE_SV-C2_706038/` | CSV | Fire hotspots |
| DeepState | `data/deepstate/wayback_snapshots/` | JSON | Territorial control |
| Equipment | `data/war-losses-data/.../russia_losses_equipment.json` | JSON | Military losses |
| Personnel | `data/war-losses-data/.../russia_losses_personnel.json` | JSON | Casualty data |
| Sentinel | `data/sentinel/sentinel_timeseries_raw.json` | JSON | Satellite metadata |

### Secondary Sources (embeddings, enrichment)
| Source | Location | Purpose |
|--------|----------|---------|
| ISW Assessments | `data/wayback_archives/isw_assessments/embeddings/` | Narrative embeddings |
| Timelines | `data/timelines/embeddings/` | Event timeline embeddings |
| Telegram | `data/telegram/embeddings/` | Social media embeddings |
| VIIRS | `data/nasa/viirs_nightlights/` | Nighttime brightness |
| HDX | `data/hdx/ukraine/` | Humanitarian data |

---

## Model Checkpoints

### Naming Conventions
```
{model_type}_{variant}_best.pt     # Best validation checkpoint
{model_type}_{variant}_final.pt    # Final training checkpoint
checkpoint_epoch_{N}.pt            # Periodic checkpoints (every 10 epochs)
```

### Standard Locations
- **HAN models**: `analysis/models/han_*.pt`
- **Unified interpolation**: `analysis/models/unified_interpolation_*.pt`
- **Multi-resolution**: `analysis/checkpoints/multi_resolution/`
- **Phase models**: `analysis/models/phase2_*.pt`, `analysis/models/phase3_*.pt`
- **Interpolation**: `analysis/models/interpolation/interp_*.pt`

### Checkpoint Contents
Full checkpoints include:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state
- `scheduler_state_dict` - LR scheduler state
- `epoch` - Training epoch
- `best_val_loss` - Best validation loss
- `history` - Training/validation history

---

## Configuration

### Environment Variables (.env)
```bash
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
VOYAGE_API_KEY=your_voyage_key
```

**Security**: Never commit `.env` with real credentials. Use `.env.example` as template.

### Training Configuration
Use `analysis/training_config.py` for:
- Data configuration (batch size, sequence length, splits)
- Model configuration (hidden dimensions, attention heads)
- Training parameters (learning rate, epochs, early stopping)

Available presets: `default`, `fast_debug`, `production`, `high_capacity`

---

## Coding Standards

### Python Style
- Use `pathlib.Path` for all path operations (not `os.path`)
- Type hints for all function signatures
- Docstrings for public functions and classes
- Maximum line length: 100 characters

### Import Order
1. Standard library
2. Third-party packages
3. Local imports (use relative imports within analysis/)

### Logging
```python
from config.logging_config import get_logger
logger = get_logger(__name__)

logger.info("Starting training...")
logger.warning("Missing data for date range")
logger.error("Model checkpoint not found", exc_info=True)
```

---

## Common Tasks

### Training a Model
```bash
cd /path/to/ML_OSINT
python -m analysis.train_han --config production
python -m analysis.train_multi_resolution --epochs 100
```

### Running Probes
```bash
python -m analysis.probes.run_probes --tier 1
```

### Generating Reports
```bash
python -m analysis.comprehensive_model_report
```

---

## Git Workflow

### Branch Strategy
- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches

### Commit Messages
```
type: short description

Longer description if needed.

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

---

## Troubleshooting

### Common Issues

**"Module not found" errors**
- Ensure you're running from project root
- Check that `PYTHONPATH` includes project root
- Use `python -m module.name` instead of `python module/name.py`

**CUDA/MPS memory errors**
- Reduce batch size in training config
- Use `PYTORCH_ENABLE_MPS_FALLBACK=1` for Mac

**Missing data files**
- Check `data/README.md` for acquisition instructions
- Run download scripts in `scripts/` directory

---

## Agent Coordination

When multiple agents work on this project:

1. **Check this file first** for project conventions
2. **Use config/paths.py** for all path references
3. **Follow the module hierarchy** - don't create circular dependencies
4. **Update documentation** when adding new modules or changing structure
5. **Run tests** before committing changes
6. **Log changes** to the appropriate log files

---

## Contacts and Resources

- **Model Documentation**: `docs/HAN.md`, `docs/Probe-specs.md`
- **Data Documentation**: `data/README.md`
- **Embeddings Guide**: `docs/multimodal-embeddings.md`, `docs/voyage.md`
- **Telegram Guide**: `docs/telegram_attribution_guide.md`
- **Documentation Index**: `docs/README.md`
