# NAME (Novelty Achievement MeasurE) Implementation Plan

## Executive Summary

This document outlines the implementation plan for extending the eval-analysis repository to support a novel evaluation metric: **NAME (Novelty Achievement MeasurE)**. This metric measures AI capability by comparing AI zero-shot performance against novice human first-attempt success rates, providing an alternative to time-based metrics.

## Metric Definition

**Core Concept**: Instead of measuring "time to complete," NAME measures "difficulty to learn" by comparing:
- **AI zero-shot performance** (single attempt, no examples)
- **Novice human first-attempt success rate** (task-specific novices, no prior experience with task)

**Primary Metric**: **N-attempt equivalence horizon**
- For each AI model, find the difficulty level where novices need N attempts to reach 50% success rate
- Analogous to "50% time horizon" but based on learning difficulty instead of time
- Example: "Claude 3.7 Sonnet has a 3-attempt equivalence" means it can solve (zero-shot) tasks that novices need 3 attempts to reach 50% success

## Implementation Phases

### Phase 1: MVP Foundation (Current Sprint)
**Goal**: Create working prototype with synthetic data that demonstrates the NAME metric calculation pipeline

#### 1.1 Data Schema Extension
Extend existing `all_runs.jsonl` format to support learning curves:

```json
{
  // Existing fields (preserved for compatibility)
  "task_id": "logic_puzzle/sudoku_easy",
  "run_id": "novice_42_attempt_3_20240101",
  "alias": "human_novice",  // or model name
  "score_binarized": 1,
  "score_cont": 1.0,

  // NEW: NAME-specific fields
  "attempt_number": 3,           // Which attempt (1-indexed)
  "prior_attempts": 2,            // Number of previous attempts
  "learner_type": "human_novice", // human_novice | human_expert | ai_zero_shot | ai_few_shot
  "examples_seen": 0,             // For AI few-shot learning (future)
  "feedback_type": "binary",      // binary | partial | full | console_output
  "feedback_received": "...",     // Actual feedback text (for console outputs)

  // NEW: Task difficulty metadata
  "first_attempt_success_rate": 0.35,  // % of novices succeeding on attempt 1
  "learning_rate_lambda": 0.42,        // Learning speed parameter λ
  "max_attempts": 10,                  // Upper bound on attempts

  // Placeholder for future expansion
  "quality_score": null,          // For graded evaluation (not binary)
  "task_domain": "logic_puzzle",  // For domain-specific analysis
  "verification_type": "automated" // automated | llm_judge | expert_review
}
```

#### 1.2 Synthetic Task Generator
Create synthetic tasks with calibrated difficulty levels:

```python
# src/generate/synthetic_tasks.py

class SyntheticTaskGenerator:
    """Generate synthetic tasks with parameterized difficulty"""

    def __init__(self):
        self.difficulty_levels = np.linspace(0.05, 0.8, 20)  # First-attempt success rates
        self.domains = ["logic_puzzle", "code_completion", "pattern_recognition",
                        "data_analysis", "creative_writing"]

    def generate_task(self, difficulty: float, domain: str) -> dict:
        """
        Generate a task definition with specified difficulty

        Args:
            difficulty: First-attempt success rate (0.0-1.0)
            domain: Task domain category

        Returns:
            Task definition with metadata
        """
        pass

    def generate_task_suite(self, n_tasks: int = 100) -> list:
        """Generate balanced suite across difficulty levels and domains"""
        pass
```

#### 1.3 Novice Learning Simulator
Implement the learning curve model:

```python
# src/simulate/novice_learning.py

class NoviceLearningSimulator:
    """Simulate novice learning curves with exponential improvement + noise"""

    def __init__(self, task_difficulty: float):
        """
        Args:
            task_difficulty: Calibrated first-attempt success rate (0.0-1.0)
        """
        self.base_rate = task_difficulty
        self.learning_gain = self._calculate_learning_gain(task_difficulty)
        self.lambda_learning = 0.4  # Default learning speed

    def _calculate_learning_gain(self, difficulty: float) -> float:
        """
        Harder tasks have more room for improvement
        Easy tasks (high base_rate) plateau quickly
        """
        return (1.0 - difficulty) * 0.8  # Cap at 80% improvement

    def probability_success(self, attempt_number: int) -> float:
        """
        P(success | n) = base_rate + learning_gain * (1 - exp(-λn)) + noise(n)

        Args:
            attempt_number: 1-indexed attempt number

        Returns:
            Probability of success on this attempt
        """
        # Core exponential learning
        learning_term = self.learning_gain * (1 - np.exp(-self.lambda_learning * attempt_number))

        # Decreasing noise (turbulent early, stable later)
        noise_std = 0.1 / np.sqrt(attempt_number)
        noise = np.random.normal(0, noise_std)

        # Clip to [0, 1]
        return np.clip(self.base_rate + learning_term + noise, 0, 1)

    def simulate_learning_trajectory(self, max_attempts: int = 10,
                                    n_learners: int = 50) -> pd.DataFrame:
        """
        Simulate multiple learners attempting the same task

        Returns:
            DataFrame with columns: learner_id, attempt_number, success
        """
        pass

    def simulate_ai_zero_shot(self, model_capability: float) -> bool:
        """
        Simulate AI zero-shot attempt

        Args:
            model_capability: AI's intrinsic capability (0.0-1.0)
                            Maps to task difficulty it can handle

        Returns:
            Success/failure on single attempt
        """
        # AI succeeds if its capability exceeds task difficulty threshold
        # Add some noise for stochasticity
        success_prob = 1 / (1 + np.exp(-10 * (model_capability - self.base_rate)))
        return np.random.random() < success_prob
```

#### 1.4 Data Generation Pipeline
Create synthetic run data:

```python
# src/generate/generate_runs.py

def generate_synthetic_runs(n_tasks: int = 100,
                           n_novices_per_task: int = 50,
                           max_attempts: int = 10,
                           models: list = None) -> pd.DataFrame:
    """
    Generate complete synthetic dataset for NAME evaluation

    Pipeline:
    1. Generate task suite with varying difficulties
    2. For each task, simulate novice learning curves (n_novices * max_attempts runs)
    3. For each task, simulate AI zero-shot attempts (n_models runs)
    4. Export to all_runs.jsonl format

    Returns:
        DataFrame matching all_runs.jsonl schema
    """
    pass
```

#### 1.5 NAME Metric Calculation
Implement core metric computation:

```python
# src/wrangle/name_metrics.py

class NAMEMetricCalculator:
    """Calculate N-attempt equivalence horizon for AI models"""

    def fit_learning_curves(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit learning curve for each task

        For each task:
        1. Extract novice attempts (attempt_number, success)
        2. Fit exponential learning model
        3. Calculate N-attempt success rates (N=1,2,3,...,10)

        Returns:
            DataFrame: task_id, first_attempt_rate, N_for_50pct, N_for_80pct, lambda, ...
        """
        pass

    def calculate_ai_performance_by_difficulty(self, runs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate AI zero-shot success rate binned by task difficulty

        Returns:
            DataFrame: model, difficulty_bin, success_rate, n_tasks, ...
        """
        pass

    def calculate_n_attempt_horizon(self, model: str,
                                   success_threshold: float = 0.5) -> dict:
        """
        Main NAME metric: Find N-attempt equivalence

        Process:
        1. For each difficulty level d, find N where P_human(success|N attempts) = 0.5
        2. For each difficulty level d, find P_AI(success|zero-shot)
        3. Find difficulty d* where P_AI(d*) = 0.5
        4. Return N(d*) as the N-attempt equivalence horizon

        Returns:
            {
                'n_attempt_horizon': 3.5,
                'confidence_interval': (2.8, 4.2),
                'equivalent_difficulty': 0.42,
                'ai_success_rate': 0.51
            }
        """
        pass

    def bootstrap_confidence_intervals(self, runs_df: pd.DataFrame,
                                      n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Bootstrap confidence intervals for NAME metric

        Hierarchical resampling:
        1. Resample tasks
        2. Within each task, resample learners
        3. Recalculate N-attempt horizon

        Returns:
            DataFrame: model, n_attempt_horizon, ci_lower, ci_upper
        """
        pass
```

#### 1.6 Visualization Tools
Create plots for NAME analysis:

```python
# src/plot/name_plots.py

def plot_learning_curves(task_id: str, runs_df: pd.DataFrame):
    """Plot: Attempt Number (x) vs Success Rate (y) for a single task"""
    pass

def plot_ai_vs_human_first_attempt(runs_df: pd.DataFrame):
    """Scatter: First-Attempt Success Rate (x) vs AI Zero-Shot Success (y)"""
    pass

def plot_n_attempt_horizon_over_time(models_df: pd.DataFrame, release_dates: pd.Series):
    """Time series: Model Release Date (x) vs N-Attempt Horizon (y)"""
    pass

def plot_difficulty_distribution(tasks_df: pd.DataFrame):
    """Histogram: Distribution of task difficulties (first-attempt success rates)"""
    pass

def plot_learning_rate_vs_difficulty(tasks_df: pd.DataFrame):
    """Scatter: Task Difficulty (x) vs Learning Rate λ (y)"""
    pass
```

#### 1.7 DVC Pipeline Integration
Add NAME stages to `dvc.yaml`:

```yaml
# New stages for NAME evaluation
generate_synthetic_tasks:
  cmd: python -m src.generate.synthetic_tasks
  deps:
    - src/generate/synthetic_tasks.py
    - params.yaml
  params:
    - name.n_tasks
    - name.difficulty_range
  outs:
    - data/external/name_tasks.yaml

generate_synthetic_runs:
  cmd: python -m src.generate.generate_runs
  deps:
    - src/generate/generate_runs.py
    - src/simulate/novice_learning.py
    - data/external/name_tasks.yaml
    - params.yaml
  params:
    - name.n_novices_per_task
    - name.max_attempts
    - name.models
  outs:
    - data/external/name_runs.jsonl

wrangle_name_metrics:
  cmd: python -m src.wrangle.name_metrics
  deps:
    - src/wrangle/name_metrics.py
    - data/external/name_runs.jsonl
    - params.yaml
  outs:
    - data/wrangled/name_learning_curves.csv
    - data/wrangled/name_horizons.csv
    - metrics/name_metrics.yaml

wrangle_name_bootstrap:
  cmd: python -m src.wrangle.name_bootstrap
  deps:
    - src/wrangle/name_metrics.py
    - data/external/name_runs.jsonl
    - params.yaml
  params:
    - name.n_bootstrap
  outs:
    - data/wrangled/name_bootstrap.csv

plot_name_results:
  cmd: python -m src.plot.name_plots
  deps:
    - src/plot/name_plots.py
    - data/wrangled/name_horizons.csv
    - data/wrangled/name_learning_curves.csv
    - params.yaml
  outs:
    - plots/name_learning_curves.png
    - plots/name_ai_vs_human.png
    - plots/name_horizon_over_time.png
```

#### 1.8 Configuration
Add NAME parameters to `params.yaml`:

```yaml
# NAME (Novelty Achievement MeasurE) Configuration
name:
  # Synthetic data generation
  n_tasks: 100
  n_novices_per_task: 50
  max_attempts: 10
  difficulty_range: [0.05, 0.8]  # First-attempt success rates

  # Learning model parameters
  default_lambda: 0.4  # Learning speed
  noise_std_initial: 0.1  # Initial noise
  max_learning_gain: 0.8  # Maximum improvement possible

  # AI models to evaluate
  models:
    - "GPT-4"
    - "Claude 3.7 Sonnet"
    - "o1"
    # Add synthetic baseline models
    - "Baseline-Weak"    # Capability: 0.3
    - "Baseline-Medium"  # Capability: 0.5
    - "Baseline-Strong"  # Capability: 0.7

  # Bootstrap configuration
  n_bootstrap: 1000
  confidence_level: 0.95

  # Metric thresholds
  success_thresholds: [0.5, 0.8]  # Calculate horizons at 50% and 80%

  # Task domains (for future expansion)
  domains:
    - "logic_puzzle"
    - "code_completion"
    - "pattern_recognition"
    - "data_analysis"
    - "creative_writing"

  # Placeholders for real data collection
  real_data:
    enabled: false
    human_recruitment_platform: null
    scoring_method: "automated"  # automated | llm_judge | expert_review
    quality_grading: false  # Enable graded scoring (not just binary)
```

### Phase 2: Real Data Infrastructure (Future)
**Goal**: Support real human data collection and non-verifiable task scoring

#### 2.1 Human Data Collection Integration
- Define protocol for recruiting task-specific novices
- Create web interface for task presentation and attempt tracking
- Implement feedback systems (console outputs, puzzle states)
- Data validation and quality checks

#### 2.2 LLM Judge Scoring
For non-verifiable tasks (creative writing, open-ended problem solving):

```python
# src/scoring/llm_judge.py

class LLMJudgeScorer:
    """Score non-verifiable tasks using LLM judges with ground truth"""

    def score_response(self, task: dict, response: str,
                      ground_truth: str, rubric: dict) -> float:
        """
        Use Claude/GPT-4 to score response against ground truth

        Returns score 0.0-1.0 with explanation
        """
        pass
```

#### 2.3 Expert Review Pipeline
For tasks requiring human expert judgment:
- Expert annotation interface
- Inter-rater reliability tracking
- Adjudication workflow for disagreements

### Phase 3: Broad Domain Expansion (Future)
**Goal**: Move beyond SWE benchmarks to everyday tasks

#### 3.1 Domain-Specific Task Suites
- **Household**: Cooking instructions, home repair, organization
- **Creative**: Writing, art critique, storytelling
- **Social**: Communication drafting, conflict resolution scenarios
- **Professional**: Email composition, meeting scheduling, data entry
- **Learning**: Tutoring problems, explanation generation

#### 3.2 Domain Adapters
Create adapters for domain-specific evaluation:
- Different feedback mechanisms per domain
- Domain-specific success criteria
- Appropriate learning curve parameterizations

#### 3.3 Economic/Social Value Weighting
Instead of `invsqrt_task_weight`, use:
- Frequency of task in daily life
- Economic value (time saved, revenue generated)
- Social value (communication, collaboration, learning)

### Phase 4: Advanced Analysis (Future)
**Goal**: Deeper insights into learning and capability

#### 4.1 Few-Shot Learning Comparison
Compare AI few-shot learning (N examples) vs human learning (N attempts)

#### 4.2 Transfer Learning
Measure how well learning on one task transfers to related tasks

#### 4.3 Failure Mode Analysis
Characterize differences between AI and human errors

#### 4.4 Skill Composition
Decompose tasks into sub-skills to understand capability profiles

## Success Criteria

### MVP Success (Phase 1)
- [ ] Synthetic data generator produces realistic learning curves
- [ ] NAME metric calculation pipeline runs end-to-end
- [ ] Visualizations clearly show AI vs human comparison
- [ ] DVC pipeline is reproducible
- [ ] Documentation enables future expansion
- [ ] Bootstrap confidence intervals validate statistical rigor

### Long-Term Success (Phases 2-4)
- [ ] Real human data collection with N>100 participants
- [ ] Task suite covers 10+ diverse domains
- [ ] LLM judge scoring validated against expert ratings (r > 0.8)
- [ ] Predictive validity: NAME metric correlates with real-world deployment success
- [ ] Published results showing NAME trends over time (doubling period, etc.)

## Scale-Up Criteria

**When to move from Phase 1 → Phase 2:**
1. MVP demonstrates clear signal in synthetic data
2. Stakeholder buy-in secured (potential users, funders)
3. Budget allocated for human data collection ($10-50K depending on scale)
4. IRB approval obtained (if required)

**When to move from Phase 2 → Phase 3:**
1. Real data collection pipeline validated (N>50 participants, 3+ task types)
2. Inter-rater reliability established (κ > 0.7)
3. Initial results published or presented
4. Community interest confirmed (citations, requests to use metric)

## Technical Debt & Future Considerations

### Known Limitations (MVP)
1. **Synthetic data realism**: Learning curves are idealized, may not match human variability
2. **Single learning model**: Exponential model may not fit all task types
3. **Domain coverage**: Starting with mostly verifiable tasks (bias toward AI strengths)
4. **No transfer learning**: Each task treated independently
5. **Binary scoring only**: Loses information about partial progress

### Migration Path
All MVP code is designed for forward compatibility:
- Data schema includes placeholders for future fields
- Scoring pipeline supports pluggable scorers (automated → LLM judge → expert)
- Domain field enables domain-specific analysis
- Quality score field ready for graded evaluation

### Performance Considerations
- Bootstrap with 1000 iterations on 100 tasks × 50 learners = manageable (~1-5 min)
- For real data (1000+ tasks), may need:
  - Distributed bootstrap computation
  - Database backend instead of JSONL
  - Incremental computation (cache intermediate results)

## Appendix: Comparison with Original Time-Horizon Metric

| Aspect | Time Horizon (Original) | NAME (New) |
|--------|------------------------|------------|
| **Core question** | How long do tasks take humans? | How hard are tasks to learn? |
| **AI measurement** | Success rate on tasks of varying duration | Zero-shot success on tasks of varying difficulty |
| **Human baseline** | Domain experts, timed completion | Task-specific novices, attempt count |
| **Main metric** | 50% time horizon (minutes) | N-attempt equivalence |
| **Task selection** | Verifiable, expert-level tasks | Everyday tasks, novice-accessible |
| **Learning captured** | No (single expert attempt) | Yes (learning curves over attempts) |
| **Bias** | Favors fast, verifiable tasks | Favors tasks with clear learning signal |
| **External validity** | "Can AI do expert work?" | "Can AI learn like humans?" |
| **Practical use** | Automation of specialized work | Automation of routine learning |

## Next Steps (Immediate)

1. Review and approve this plan
2. Implement synthetic task generator
3. Implement novice learning simulator
4. Generate initial synthetic dataset
5. Implement NAME metric calculation
6. Create basic visualizations
7. Integrate into DVC pipeline
8. Document and iterate

---

**Document Status**: Draft v1.0
**Author**: Claude (AI Assistant)
**Date**: 2025-11-24
**Next Review**: After MVP implementation
