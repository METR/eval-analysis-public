# Measuring AI Ability to Complete Long Tasks

This is the code for the paper Measuring AI Ability to Complete Long Tasks. 

Despite rapid progress on AI benchmarks, the real-world meaning of benchmark
performance remains unclear. To quantify the capabilities of AI systems in terms
of human capabilities, we propose a new metric: 50%-task-completion time hori-
zon. This is the time humans typically take to complete tasks that AI models can
complete with 50% success rate. We first timed humans with relevant domain ex-
pertise on a combination of RE-Bench, HCAST, and 66 novel shorter tasks. On
these tasks, current frontier AI models such as Claude 3.7 Sonnet have a 50% time
horizon of around 50 minutes. Furthermore, frontier AI time horizon has been
doubling approximately every seven months since 2019, though the trend may
have accelerated in 2024. The increase in AI models’ time horizons seems to be
primarily driven by greater reliability and ability to adapt to mistakes, combined
with better logical reasoning and tool use capabilities. We discuss the limitations
of our results—including their degree of external validity—and the implications
of increased autonomy for dangerous capabilities. If these results generalize to
real-world software tasks, extrapolation of this trend predicts that within 5 years,
AI systems will be capable of automating many software tasks that currently take
humans a month.

## Installation

This project contains a dev container, which we recommend using.  Alternatively, you can view the
[.devcontainer/Dockerfile](Dockerfile) to see which dependencies need to be installed.

 After installing those dependencies, the figures can be recreated by running:

```
poetry install
poetry run dvc repro
```

An example of additional analysis which can be performed after completing these steps can be found at
[example_analysis.ipynb](example_analysis.ipynb)