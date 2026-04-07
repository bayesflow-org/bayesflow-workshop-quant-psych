# Copilot instructions

## Amortized inference workflow
- For simulation-based inference, amortized Bayesian inference, neural posterior estimation, neural likelihood estimation, neural ratio estimation, posterior amortization, and BayesFlow workflows, prefer the `amortized-workflow` skill if available.
- Use the amortized-workflow skill for model setup, simulator assumptions, training, validation, calibration, recovery, posterior contraction, and reporting.
- Do not present an amortized inference result without checking calibration and parameter recovery when simulated ground truth is available.
- Prefer concise, reproducible code and explicit assumptions.

## Project defaults
- Always ask the user if there is an existing bayesflow conda environment.
- Prefer BayesFlow/PyTorch/JAX tools already present in that environment.
- Keep outputs structured: assumptions, implementation, diagnostics, limitations.