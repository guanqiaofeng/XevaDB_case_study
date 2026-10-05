import sys
from pathlib import Path

from pipeline_utils import make_case1_paths, run_growth_ml_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

RANDOM_STATE = 42

# RandomForest and LassoedForest each independently select their own
# max_depth/min_samples_leaf via a nested inner-CV search over this grid,
# per outer fold.
RF_MAX_DEPTH_GRID = (3, 6)
RF_MIN_SAMPLES_LEAF_GRID = (1, 2)

# ElasticNet's C/l1_ratio, nested-tuned per outer fold via inner_cv.
EN_L1_RATIOS = (0.1, 0.5, 0.9)
EN_CS = (0.01, 0.1, 1.0, 10.0)

# LassoedForest's own ridge/lasso post-selection C, nested-tuned per outer
# fold via inner_cv.
LF_CS = (0.001, 0.01, 0.1, 1.0, 10.0, 100)

paths = make_case1_paths()

run_growth_ml_pipeline(
    cohort="lung",
    paths=paths,
    n_splits=3,
    n_hvg=600,
    rf_max_depth_grid=RF_MAX_DEPTH_GRID,
    rf_min_samples_leaf_grid=RF_MIN_SAMPLES_LEAF_GRID,
    en_l1_ratios=EN_L1_RATIOS,
    en_cs=EN_CS,
    lf_cs=LF_CS,
    random_state=RANDOM_STATE,
)
