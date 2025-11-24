# NAME (Novelty Achievement MeasurE) - MVP Implementation

## Overview

NAME is a novel AI evaluation metric that measures AI capability by comparing AI zero-shot performance against novice human learning curves. Unlike time-based metrics (METR's time-horizon) or static benchmarks (MMLU, etc.), NAME asks: **"Can AI solve on first try what novices learn over multiple attempts?"**

## Core Innovation

**N-Attempt Equivalence Horizon**: The main NAME metric indicating task difficulty (measured by attempts novices need) where AI achieves target success rate.

**Example**: "Claude 3.7 Sonnet has a 3-attempt equivalence horizon" means it can zero-shot solve tasks that novices need 3 attempts to reach 50% success.

## What Makes NAME Unique?

1. **Novice-Centric Baseline**: Compares to novice first-attempt, not expert performance
2. **Learning Curve Comparison**: Captures full learning trajectories, not single-shot evaluation
3. **Everyday Task Focus**: Targets common tasks accessible to novices, not expert-level challenges
4. **Interpretable Metric**: N-attempt equivalence is intuitive and trackable over time

See `docs/NAME_IMPLEMENTATION_PLAN.md` for detailed comparison with existing benchmarks.

## MVP Architecture

### Data Pipeline

```
Synthetic Task Generator
    ↓ (100 tasks, 5 domains)
Novice Learning Simulator
    ↓ (50 learners × 10 attempts)
AI Zero-Shot Simulator
    ↓ (3 models × 20 attempts)
56,000 Synthetic Runs
    ↓
Learning Curve Fitter
    ↓
N-Attempt Horizon Calculator
    ↓
Visualizations & Metrics
```

### File Structure

```
src/
├── simulate/
│   └── novice_learning.py      # Exponential learning model
├── generate/
│   ├── synthetic_tasks.py      # Task generator (5 domains)
│   └── generate_runs.py        # Run data generator
├── wrangle/
│   └── name_metrics.py         # Learning curve fitting, horizon calculation
└── plot/
    └── name_plots.py           # Visualization suite

data/
├── external/
│   ├── name_tasks.yaml         # 100 synthetic tasks
│   └── name_runs.jsonl         # 56,000 runs
└── wrangled/
    ├── name_learning_curves.csv # 100 fitted curves
    └── name_horizons.csv        # 3 model horizons

plots/
├── name_learning_curves_sample.png      # Sample learning trajectories
├── name_ai_vs_human_first_attempt.png   # Zero-shot vs first-attempt
├── name_n_attempt_horizons.png          # Main NAME metric
└── name_difficulty_distribution.png     # Task difficulty histogram
```

## Running the NAME Pipeline

### Quick Start

```bash
# Run complete pipeline via DVC
poetry run dvc repro plot_name_results

# Or run individual stages
poetry run python -m src.generate.synthetic_tasks
poetry run python -m src.generate.generate_runs
poetry run python -m src.wrangle.name_metrics
poetry run python -m src.plot.name_plots
```

### What Gets Generated

1. **100 synthetic tasks** (20 per domain):
   - logic_puzzle (verifiable)
   - code_completion (verifiable)
   - pattern_recognition (verifiable)
   - creative_writing (non-verifiable)
   - communication_tasks (non-verifiable)

2. **56,000 runs**:
   - 50,000 novice runs (50 learners × 10 attempts × 100 tasks)
   - 6,000 AI runs (3 models × 20 attempts × 100 tasks)

3. **Learning curves**: Exponential fit for each task
4. **NAME horizons**: N-attempt equivalence for each AI model
5. **4 visualization plots**: Learning curves, comparisons, distributions

## Key Results (Synthetic Data)

| Model | N-Attempt Horizon (50%) | N-Attempt Horizon (80%) | Overall Success |
|-------|------------------------|------------------------|----------------|
| baseline-weak | 1.40 attempts | 7.59 attempts | 25.4% |
| baseline-medium | 0.77 attempts | 5.28 attempts | 60.4% |
| baseline-strong | 1.00 attempts | 4.00 attempts | 87.4% |

**Interpretation**:
- **weak**: Performs below novice first-attempt on average
- **medium**: Near novice first-attempt level
- **strong**: Matches novice first-attempt performance

## Learning Curve Model

**Noisy Exponential Learning**:
```
P(success | attempt n) = base_rate + learning_gain × (1 - e^(-λn)) + noise(n)
```

**Parameters**:
- `base_rate`: First-attempt success (task difficulty)
- `learning_gain`: Maximum learnable improvement
- `λ` (lambda): Learning speed (typically 0.2-0.6)
- `noise(n) ~ N(0, σ₀/√n)`: Decreasing variance

**Why this model?**
- Captures semi-random start, turbulent improvement, steady plateau
- Mathematically tractable for N-attempt calculation
- Psychologically plausible (matches human learning research)
- Simple (3 parameters vs logistic's 4)

## Task Domains

### Verifiable Tasks (60%)
Automated scoring possible:
- **logic_puzzle**: Sudoku, pattern completion, deduction
- **code_completion**: Fill-in code, debugging, simple algorithms
- **pattern_recognition**: Sequence prediction, visual patterns

### Non-Verifiable Tasks (40%)
Require LLM judge or expert review:
- **creative_writing**: Short stories, poetry, descriptions
- **communication_tasks**: Email drafting, explanations, persuasive writing

## Scale-Up Plan

### Phase 2: Real Human Data
1. Recruit task-specific novices (N=100+ per task type)
2. Track learning over 10 attempts with binary feedback
3. Validate learning curve models against real data
4. Refine task selection based on human performance

**Criteria to proceed**: MVP demonstrates clear signal in synthetic data ✓

### Phase 3: Broad Domain Expansion
- Household tasks (cooking, home repair, organization)
- Creative tasks (art, music, storytelling)
- Social tasks (communication, conflict resolution)
- Professional tasks (data entry, scheduling, email)
- Learning tasks (tutoring, explanation)

**Criteria to proceed**: Real data collection validated (N>50 participants)

### Phase 4: Advanced Analysis
- Few-shot learning comparison
- Transfer learning measurement
- Failure mode analysis
- Skill composition decomposition

**Criteria to proceed**: Initial results published/presented

## Extending the MVP

### Adding Real AI Models

1. Create adapter in `src/simulate/novice_learning.py`:
```python
def create_real_ai_model(model_name: str, api_key: str):
    """Adapter for real AI model APIs."""
    # Implement API calls to GPT-4, Claude, etc.
    pass
```

2. Update `generate_runs.py` to use real models
3. Run pipeline: `dvc repro plot_name_results`

### Adding Real Tasks

1. Define tasks in YAML format:
```yaml
tasks:
  - task_id: "real_task_001"
    domain: "cooking"
    description: "Soft boil an egg"
    verification_type: "expert_review"
```

2. Replace `generate_name_tasks` stage in `dvc.yaml`
3. Collect human data matching schema in `name_runs.jsonl`

### Adding New Domains

1. Add domain to `TASK_DOMAINS` in `synthetic_tasks.py`
2. Define verification method in `domain_verification` dict
3. Update color scheme in `name_plots.py` (optional)

## Technical Details

### Data Schema

Each run in `name_runs.jsonl`:
```json
{
  "task_id": "logic_puzzle/task_0042",
  "run_id": "novice_0023_attempt_05_...",
  "alias": "human_novice" | "model_name",
  "score_binarized": 0 | 1,
  "attempt_number": 5,
  "learner_type": "human_novice" | "ai_zero_shot",
  "first_attempt_success_rate": 0.35,
  "learning_rate_lambda": 0.42,
  "task_domain": "logic_puzzle",
  "verification_type": "automated",
  ...
}
```

### Metric Calculation

1. **Fit learning curves** (scipy.optimize.curve_fit):
   - Per-task exponential model
   - Solves for base_rate, learning_gain, λ
   - Calculates N for 50% and 80% success

2. **Calculate AI performance**:
   - Bin tasks by difficulty
   - Average AI success per bin
   - Interpolate to find difficulty @ 50%/80% AI success

3. **Find N-attempt horizon**:
   - Interpolate N(difficulty) curve
   - Evaluate at AI's 50%/80% difficulty
   - Result: N-attempt equivalence

### Bootstrap Confidence Intervals

Future implementation (not in MVP):
```python
# Hierarchical resampling: task → learner → attempt
for i in range(1000):
    resample_tasks()
    resample_learners_within_tasks()
    recalculate_horizons()
quantiles = np.percentile(horizons, [2.5, 97.5])
```

## Comparison with Time-Horizon Metric

| Aspect | Time Horizon | NAME |
|--------|-------------|------|
| **Question** | "How long do experts take?" | "How hard to learn?" |
| **Baseline** | Domain experts | Task-specific novices |
| **Measurement** | Completion time (minutes) | Attempts to proficiency |
| **Task bias** | Verifiable, expert tasks | Everyday, novice tasks |
| **Learning** | No (single attempt) | Yes (curves over attempts) |
| **External validity** | "Can AI do expert work?" | "Can AI learn like humans?" |

Both metrics are valuable and complementary. NAME may better predict consumer AI adoption.

## Known Limitations (MVP)

1. **Synthetic data**: Learning curves are idealized, not validated against humans
2. **Single learning model**: May not fit all task types (some need logistic, power law)
3. **No bootstrap CI**: Confidence intervals not yet implemented
4. **Domain coverage**: Limited to 5 domains, missing many everyday tasks
5. **Binary scoring**: No partial credit or quality grading
6. **Static difficulty**: Doesn't account for skill transfer across tasks

## Next Steps

1. ✅ Complete MVP with synthetic data
2. ⬜ Collect pilot human data (N=20 participants, 10 tasks)
3. ⬜ Validate learning curve model against real data
4. ⬜ Implement bootstrap confidence intervals
5. ⬜ Add LLM judge scoring for non-verifiable tasks
6. ⬜ Expand task suite to 10+ domains
7. ⬜ Test with real AI models (GPT-4, Claude, etc.)
8. ⬜ Publish results and comparison with time-horizon

## Citation

If you use NAME in your research, please cite:

```bibtex
@software{name2025,
  title={NAME: Novelty Achievement MeasurE - A Learning-Based AI Evaluation Framework},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/eval-analysis-public_exp}
}
```

## Contact & Contributing

- **Issues**: GitHub Issues
- **Questions**: [Your contact]
- **Contributions**: PRs welcome! See `docs/NAME_IMPLEMENTATION_PLAN.md` for architecture details

## License

[Same as parent repository]

---

**Last Updated**: 2025-11-24
**Version**: MVP 1.0
**Status**: Synthetic data demonstration complete ✓
