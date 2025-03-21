# Horizon Length Report

## Links

- [Overleaf][2]
- [Methodology FAQ][1]
- [Standup Notes][3]

## Methodology

Our estimate for the horizon length growth curve has three steps:

1. Get a human time estimate for each task in the dataset.
2. Use logistic regression on the success rate of each model on each task to estimate the "horizon length" of each model-- the time at which the model succeeds at 50% of tasks that take humans that long.
3. Use linear regression on the horizon lengths to estimate the doubling time of the horizon length over model releases.

### Human time estimates

Currently we average together the times that successful human baseliners took to complete each task. If there is no baseline, we use a time estimate created by a METR employee.

### Horizon length

We use logistic regression to predict the success rate of each model on a task as a function of log(human time).

### Doubling time

We use linear regression to predict the doubling time of the horizon length over model releases.

### Uncertainty analysis

Error bars are calculated by bootstrapping our dataset of runs and carrying the analysis forward. Tasks may be correlated, so we use hierarchical bootstrapping, where we first sample task families, then tasks within those families, then individual runs. Plots are in `plots/bootstrap/`.

There is a long list of factors that could affect our error bars, see the [FAQ][1].

[1]: https://docs.google.com/document/d/15MzV2YT2BFu2PxM08bhEY3jRVwj2WDGMmjCHxs0BgfE/edit?tab=t.0
[2]: https://www.overleaf.com/project/67b50496c4be7856b48acc00
[3]: https://docs.google.com/document/d/1Sx70CZbfSu1HMX-x-YfuqpIq9Lk0hmUnpgYupQ-K97k/edit