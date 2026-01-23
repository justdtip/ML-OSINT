# Omitted Variable Analysis Pipeline
# Stage modules for systematic identification of missing data sources

import sys
from pathlib import Path

# Ensure analysis directory is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from .residual_extraction import extract_residuals, ResidualExtractionResults
from .temporal_analysis import run_temporal_analysis, TemporalAnalysisResults
from .factor_extraction import run_factor_extraction, FactorExtractionResults
from .candidate_correlation import run_candidate_correlation, CandidateCorrelationResults
from .granger_causality import run_granger_causality, GrangerCausalityResults
from .ranking_report import run_ranking_report, RankingResults
