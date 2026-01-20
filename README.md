# METR Time Horizon Analysis

This repository contains the analysis code and data for METR's time horizon methodology, as described in ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499).

## Overview

The time horizon methodology measures AI agent capabilities by:
1. Collecting tasks with known **human completion times**
2. Running AI agents on these tasks and recording success/failure
3. Fitting a **logistic curve** modeling P(success) as a function of log2(human_minutes)
4. Extracting the **"time horizon"** - the task duration where the model hits a success threshold

**Key finding**: AI agent time horizons have been doubling approximately every 7 months.

## Repository Structure

```
.
├── src/horizon/           # Analysis code (installable Python package)
│   ├── utils/             # Core utilities (logistic regression, plots)
│   ├── wrangle/           # Data wrangling (bootstrap, logistic fitting)
│   └── plot/              # Plot generation modules
├── data/
│   └── external/
│       └── release_dates.yaml  # Model release dates
└── reports/
    ├── time-horizon-1-0/  # Time Horizon v1.0
    │   ├── dvc.yaml       # DVC pipeline definition
    │   ├── params.yaml    # Report parameters
    │   └── data/raw/
    │       └── runs.jsonl # Run data
    └── time-horizon-1-1/  # Time Horizon v1.1
        ├── dvc.yaml
        ├── params.yaml
        └── data/raw/
            └── runs.jsonl
```

## Installation

```bash
# Clone the repository
git clone https://github.com/METR/eval-analysis-public.git
cd eval-analysis-public

# Install the horizon package in editable mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Running the Reports

Each report has its own DVC pipeline. To run a report:

```bash
# Run the time-horizon-1-0 report
cd reports/time-horizon-1-0
dvc repro

# Run the time-horizon-1-1 report
cd reports/time-horizon-1-1
dvc repro
```

The pipelines will:
1. Run bootstrap sampling for confidence intervals
2. Fit logistic regression models
3. Generate plots and metrics

## Data Format

The `runs.jsonl` files contain one JSON object per line with the following key fields:

| Field | Description |
|-------|-------------|
| `task_id` | Unique task identifier |
| `task_family` | Group of related tasks |
| `alias` | Public model name |
| `score_binarized` | 0 (failure) or 1 (success) |
| `score_cont` | Continuous score 0-1 |
| `human_minutes` | How long a qualified human expert takes |
| `invsqrt_task_weight` | Diversity-adjusted weight for this run |

## Key Outputs

After running `dvc repro`, you'll find:

- `data/wrangled/bootstrap/*.csv` - Bootstrap samples for confidence intervals
- `data/wrangled/logistic_fits/*.csv` - Logistic regression fits with p50/p80 horizons
- `plots/` - Generated visualizations
- `metrics/` - YAML files with key metrics

## Analyzing Results

```python
import pandas as pd

# Load logistic fits
fits = pd.read_csv("reports/time-horizon-1-0/data/wrangled/logistic_fits/headline.csv")

# See horizons by agent
print(fits[["agent", "p50", "p50q0.025", "p50q0.975"]].sort_values("p50", ascending=False))

# Load raw runs
runs = pd.read_json("reports/time-horizon-1-0/data/raw/runs.jsonl", lines=True)

# Success rate by agent
print(runs.groupby("alias")["score_binarized"].mean().sort_values(ascending=False))
```

## Reports

### time-horizon-1-0
The main model report with comprehensive analysis of 48+ models using the original metr-task-standard evaluation framework, including:
- Time horizon trends (p50, p80)
- Bootstrap confidence intervals
- Token usage analysis
- Comparison overlays with time-horizon-1-1 results

### time-horizon-1-1
Results similar to above, but run on an updated task suite. Includes stages for comparing doubling times with time-horizon-1-0 (`compare_doubling_times_vs_th_1_0`).

## Citation

If you use this code or data, please cite:

```bibtex
@article{metr2025horizon,
  title={Measuring AI Ability to Complete Long Tasks},
  author={METR},
  journal={arXiv preprint arXiv:2503.14499},
  year={2025}
}
```

## License

See LICENSE file for details.
