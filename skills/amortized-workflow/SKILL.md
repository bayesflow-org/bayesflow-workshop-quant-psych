---
name: amortized-workflow
description: >
  Opinionated amortized Bayesian workflow with BayesFlow for simulation-based inference (SBI).
  Contains critical guardrails that agents will usually not apply unprompted — always consult
  before writing BayesFlow code. Trigger on: simulation-based inference, amortized inference, approximate Bayesian inference,
  BayesFlow, neural posterior estimation, posterior amortization, simulator design, prior design
  for SBI, offline/online simulation pipelines,
  uncertainty quantification from simulators, structured data encoders (sets, time series, images),
  or mentions of BasicWorkflow, fit_online, fit_offline, fit_disk, FlowMatching, DiffusionModel,
  StableConsistencyModel, summary networks, adapters, or simulation budgets.
license: MIT
metadata:
  author: [Stefan T. Radev](https://bayesops.com)
  version: "0.0.1"
---

# Amortized Bayesian Workflow

## Workflow overview

Every amortized Bayesian analysis follows this sequence. Do not skip steps — especially simulator validation and model criticism.

1. **Formulate** — Define the generative story. What latent variables or parameters generated the observations?
2. **Specify the simulator regime** — Decide whether inference will rely on:
   - **Online training**: simulator available and fast enough to generate data on the fly
   - **Offline training**: pre-simulated data available and fits in memory
   - **Disk training**: pre-simulated data available on disk and too large for memory
3. **Define prior + observation model or simulation bank**
   - If online: implement prior and observation model and wrap them in a simulator
   - If offline/disk: ensure simulations are already generated from the intended prior and data-generating process or need pre-processing
4. **Choose the architecture** — this step is critical; getting it wrong silently ruins inference. See `references/conditioning.md` for the full conditioning logic and decision table.
   - **"Simple vector"** means the observation is a **single fixed-length feature vector** whose element order is meaningful (e.g., 5 named sensor readings, a pre-computed summary statistic). Only then: route through `inference_conditions` with no summary network.
   - **Set-based / exchangeable data** — If the simulator produces **N observations that are exchangeable** (i.e., their joint likelihood is invariant to permutation), the data is a **set**, not a vector. This includes: N i.i.d. draws, regression datasets with (x, y) pairs, repeated measurements, trial-level data, cross-sectional samples. Route through `summary_variables` with a `SetTransformer`. **Never put this in `inference_conditions`.**
   - **Time series** — ordered sequences: route through `summary_variables` with `TimeSeriesTransformer` or `TimeSeriesNetwork`.
  - **Images as conditions / observations for parameter inference** — route through `summary_variables` with `ConvolutionalNetwork`.
  - **Images as inferential targets** — conditional image generation, spatial field generation, denoising, and other image-valued outputs require an **image-capable diffusion inference network**. Use `bf.networks.DiffusionModel(subnet=...)` with `UNet`, `UViT`, or `ResidualUViT`; see `references/image-generation.md`.
   - **A workflow can use both slots simultaneously.** Fixed-length metadata (e.g., sample size N, scalar design variables) can go in `inference_conditions` while structured observations go in `summary_variables`.
   - **When in doubt, use a summary network.** It is always safer to include one than to omit one; a summary network will always be needed if the data has more than one axis.
5. **Build the workflow** — Prefer `bf.BasicWorkflow(...)`
6. **Run simulation sanity checks** — Before training, verify that simulated data look plausible and span the relevant range of real observations
7. **Train the amortizer**
   - `workflow.fit_online(...)` if generating on the fly
   - `workflow.fit_offline(...)` if loading all simulations in memory
   - `workflow.fit_disk(...)` if streaming simulations from disk
8. **Diagnose in silico** — Use held-out simulations with known ground truth using the workflow's built-in diagnostics: `workflow.compute_default_diagnostics(...)` for numerical results and `workflow.plot_default_diagnostics(...)` for visual diagnostics.
9. **Amortized inference on real data** — Use `workflow.sample(...)`
10. **Posterior predictive checks (PPCs)** — Re-simulate data from posterior samples and compare to the real data using model-specific test quantities
11. **Report results** — Include simulator assumptions, training regime, simulation budget, diagnostic performance, PPC results, and limitations

## Hard rules — MUST and NEVER

These rules are non-negotiable. Violating any of them will silently produce wrong results.

- **MUST use `bf.Adapter()` for data routing.** Build an explicit adapter chain with `.as_set()`, `.constrain()`, `.concatenate()`, etc. and pass `adapter=` to `BasicWorkflow`. Do NOT invent custom adapter functions, lambdas, or manual preprocessing — the adapter handles training and inference identically. The naming shorthand (`inference_variables=`, `summary_variables=` as kwargs) is ONLY acceptable when the simulator output already has the exact shapes/dtypes the networks expect AND no parameter has bounded support. When in doubt, use an explicit adapter.
- **MUST start with the Small network configuration** from `references/model-sizes.md`. Scale up to Base or Large ONLY if diagnostics show poor recovery or calibration after sufficient training. Oversized networks waste compute and can hurt calibration on simple problems.
- **MUST use `workflow.simulate(N)` to generate test data** for diagnostics — not a Python for-loop over `simulator()`. The simulator returned by `bf.make_simulator` is a batched object; `workflow.simulate(N)` calls it efficiently and returns data in the format the workflow expects.
- **MUST use `workflow.compute_default_diagnostics(test_data=...)` and `workflow.plot_default_diagnostics(test_data=...)`** for in-silico diagnostics. NEVER hand-roll coverage, bias, or calibration computations — the built-in methods are correct, complete, and consistent with the house thresholds.
- **For image-valued inference targets, follow `references/image-generation.md`.** MUST use `bf.networks.DiffusionModel(subnet=...)` with `UNet`, `UViT`, or `ResidualUViT` — not the default low-dimensional setup. Conditions must be spatially concatenable with the image target (broadcast `(B, D)` to `(B, H, W, D)`). `scripts/check_diagnostics.py` does not apply; use visual sample grids instead.
- **`workflow.sample()` returns original parameter names, NOT `"inference_variables"`.** The adapter's reverse transform restores the original keys from the simulator (e.g., `"alpha"`, `"beta"`, `"sigma"`). Each parameter has shape `(batch, num_samples)` for scalars or `(batch, num_samples, d)` for vectors. NEVER index into `"inference_variables"` — that key does not exist in the output.
- **MUST reuse the existing simulator functions for PPCs.** NEVER re-implement the generative model by hand for posterior predictive checks. Loop over a subset of posterior draws (50 is a good default), indexing over the `num_samples` axis, and pass each draw through the simulator's forward model.
- **MUST save `history.history` as JSON** (not CSV, not a DataFrame — it is a plain dict). Then run `scripts/inspect_training.py` or call `inspect_history()` in-process.
- **MUST pass `validation_data=` to all `fit_*` calls.** Use an integer (e.g., `validation_data=300`) for online training.
- **NEVER mix an explicit `adapter=` with the naming shorthand** (`inference_variables=`, `summary_variables=`, `inference_conditions=` as kwargs). They are mutually exclusive. Passing both causes silent conflicts.
- **NEVER flatten structured data into `inference_conditions`.** Sets, time series, and images MUST go through `summary_variables` with an appropriate summary network.
- **`workflow.plot_default_diagnostics()` ALWAYS returns a `dict[str, Figure]`.** Iterate directly over `.items()` to save figures. Do not type-check or branch on the return type.
- **NEVER skip in-silico diagnostics.** Good training loss does not imply good inference.

## Installation

Install BayesFlow and a backend. Prefer JAX unless the user has a strong reason to use PyTorch or TensorFlow.

```bash
pip install "bayesflow"
```

## BayesFlow workflow template

```python
import bayesflow as bf
import numpy as np

RANDOM_SEED = sum(map(ord, "my-sbi-analysis-v1"))
rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------
# 1. Define prior + observation model (online case)
# --------------------------------------------------

def my_prior():
    theta = ...
    return {"parameters": theta}

def my_observation_model(parameters):
    x = ...
    return {"observables": x}

# bf.make_simulator returns a BATCHED simulator object.
# Use simulator.sample(batch_size) or workflow.simulate(N) — NEVER loop with simulator() in Python.
simulator = bf.make_simulator([my_prior, my_observation_model])

# --------------------------------------------------
# 2. Choose architecture
# --------------------------------------------------

# IMPORTANT: Choose summary network based on data structure.
# - None ONLY if the observation is a single fixed-length vector with meaningful element order.
# - SetTransformer if the data consists of N exchangeable observations (e.g., i.i.d. draws,
#   regression datasets, repeated measurements). This is the MOST COMMON case.
# - TimeSeriesTransformer / TimeSeriesNetwork for ordered sequences.
# - ConvolutionalNetwork for image data when the IMAGE is the condition / observation and the target is low-dimensional.
# - If the TARGET itself is an image or spatial field, switch to the image-generation workflow in
#   references/image-generation.md and use DiffusionModel(subnet=UNet/UViT/ResidualUViT).
#
# ALWAYS start with the SMALL config from references/model-sizes.md. Scale up only if diagnostics
# show poor recovery or calibration after sufficient training.
summary_net = bf.networks.SetTransformer(...)  # see references/model-sizes.md — start with Small

inference_net = bf.networks.FlowMatching()
# alternatives:
# bf.networks.DiffusionModel()
# bf.networks.StableConsistencyModel()   # when fast inference is especially important

# --------------------------------------------------
# 3. Build the adapter (MUST use bf.Adapter)
# --------------------------------------------------

# See references/adapter.md for the full API and a step-by-step example.
# The adapter routes simulator output to the correct network slots and handles
# parameter constraints, set assembly, dtype conversion, and concatenation.
#
# NEVER mix adapter= with naming kwargs (inference_variables=, summary_variables=, etc.).
# NEVER write a custom adapter function — always use the bf.Adapter() chain.
adapter = (
    bf.Adapter()
    .as_set(["observables"])              # (N,) -> (N, 1) for SetTransformer
    .constrain("parameters", lower=0)     # if parameters have bounded support
    .convert_dtype("float64", "float32")
    .concatenate(["observables"], into="summary_variables")
    .concatenate(["parameters"], into="inference_variables")
)

# --------------------------------------------------
# 4. Create workflow
# --------------------------------------------------

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,
    checkpoint_filepath="checkpoints",
    checkpoint_name="your_model",
)

# --------------------------------------------------
# 5. Train (use deep learning best practices)
# --------------------------------------------------

history = workflow.fit_online(
    epochs=500,
    batch_size=32,
    num_batches_per_epoch=100,
    validation_data=300,  # auto-simulates 300 validation sets
)

# --- Mandatory: save history and inspect training convergence ---
import json
from scripts.inspect_training import inspect_history

with open("history.json", "w") as f:
    json.dump(history.history, f)

training_report = inspect_history(history.history)
print(json.dumps(training_report, indent=2))

if not training_report["overall"]["ok"]:
    print("TRAINING ISSUES — address before continuing:")
    for issue in training_report["overall"]["issues"]:
        print(f"  - {issue}")

# --------------------------------------------------
# 6. In-silico diagnostics on held-out simulations
# --------------------------------------------------

# MUST use workflow.simulate() — NEVER loop over simulator() manually.
# MUST use the built-in diagnostics — NEVER hand-roll coverage/bias/calibration.
test_data = workflow.simulate(300)

# plot_default_diagnostics ALWAYS returns a dict[str, Figure].
# No need to type-check — iterate directly.
figures = workflow.plot_default_diagnostics(test_data=test_data)
for name, fig in figures.items():
    fig.savefig(f"diagnostics_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)

# --- Mandatory: save diagnostics and check house thresholds ---
from scripts.check_diagnostics import check_diagnostics

metrics.to_csv("metrics.csv")
print(metrics)

diag_report = check_diagnostics(metrics)
print(json.dumps(diag_report, indent=2))

if diag_report["overall"]["decision"] == "STOP":
    raise RuntimeError(diag_report["overall"]["recommendation"])

# --------------------------------------------------
# 7. Amortized inference on real data (if any)
# --------------------------------------------------

# The adapter is applied in REVERSE after sampling: the returned dict
# contains the ORIGINAL parameter names from the simulator (e.g.,
# "alpha", "beta", "sigma"), NOT "inference_variables".
# Each parameter has shape (batch, num_samples) for scalars or
# (batch, num_samples, d) for vectors.
real_data = {"observables": x_obs}
samples = workflow.sample(
    conditions=real_data,
    num_samples=1000
)
# e.g. samples["alpha"].shape == (1, 1000, 1)
#      samples["beta"].shape  == (1, 1000, 1)

# --------------------------------------------------
# 8. Posterior predictive checks (custom)
# --------------------------------------------------

# MUST reuse the existing simulator functions for PPCs.
# NEVER re-implement the generative model by hand.
# Loop over a subset of posterior draws (50 is a good default),
# indexing over the num_samples axis:
#
# n_ppc = 50
# for s in range(n_ppc):
#     # Extract single draw (index over num_samples dim)
#     theta_s = {k: samples[k][0, s] for k in ["alpha", "beta", ...]}
#     # Pass through the simulator's forward model
#     x_rep = my_observation_model(**theta_s)
#     # Compare x_rep to x_obs using domain-specific summaries
```

## Offline and disk training

Only **online training** requires a callable simulator.

If the simulator is slow, unavailable, proprietary, or external, BayesFlow can still be trained from simulations alone.

### Offline training

```python
workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(),
    summary_network=summary_net,
    inference_variables=["parameters"],
    summary_variables=["observables"],  # NOT inference_conditions — structured data needs a summary network
    ...
)

history = workflow.fit_offline(data=simulated_data, epochs=500, batch_size=32, validation_data=validation_data)
```

### Disk training

```python
def custom_load(path_to_file):
    # load file + preprocessing
    return {"observables": x, "parameters": params}


workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(),
    summary_network=summary_net,
    inference_variables=["parameters"],
    summary_variables=["observables"],  # NOT inference_conditions — structured data needs a summary network
    ...
)

history = workflow.fit_disk(root="path/to/simulation_bank", load_fn=custom_load, epochs=500, batch_size=32, validation_data=validation_data)
```

## Architecture defaults

### Adapters

See `references/adapter.md` for the full adapter API and a step-by-step example.

When `BasicWorkflow` receives `inference_variables=`, `summary_variables=`, etc. as keyword arguments, it constructs a **minimal implicit adapter** that only renames and routes those keys. This is sufficient when the simulator output already has the right shapes and dtypes. Use an explicit `bf.Adapter()` chain and pass `adapter=` to the workflow whenever you need structural transforms (`.as_set`, `.broadcast`), parameter constraints (`.constrain`), feature engineering (`.sqrt`, `.log`), dtype coercion (`.convert_dtype`), or custom concatenation. The explicit adapter and the naming shorthand are mutually exclusive — do not use both.

### Summary networks

See `references/conditioning.md` for the full conditioning model `p(inference_variables | summary_net(summary_variables), inference_conditions)` and a decision table.

Use a summary network (and route data through `summary_variables`) whenever observations are not a single fixed-length vector with meaningful element order. **Most statistical models produce set-based (exchangeable) data** — N i.i.d. draws, regression datasets, repeated measurements, cross-sectional samples. These MUST use a `SetTransformer` via `summary_variables`, never be flattened and placed in `inference_conditions`.

- **Set-based / exchangeable data (most common case)**:
  - `bf.networks.SetTransformer` — **required default** for any model that generates N exchangeable observations
  - `bf.networks.DeepSet` — simpler baseline (discouraged)
- **Images as conditions / observations for parameter inference**:
  - `bf.networks.ConvolutionalNetwork` - extensible default
- **Time series**:
  - `bf.networks.TimeSeriesNetwork` — simplest default
  - `bf.networks.TimeSeriesTransformer` — stronger sequence model
  - `bf.networks.FusionTransformer` — for more complex sequential structure
As a heuristic, always start by setting the `summary_dim` argument to 2x the number of parameters to be estimated.

- **Custom summary networks** — Only when the user explicitly requests one OR the data modality is non-typical (not images, time series, sets, or vectors). See `references/custom-summary.md`.

### Inference networks

For most posterior approximation tasks, default to one of:

- `bf.networks.FlowMatching()`
- `bf.networks.DiffusionModel()`
- `bf.networks.StableConsistencyModel()`

Use **StableConsistencyModel** when very fast posterior sampling at deployment time is especially important.

### Image-valued inference targets

If the inferential target is an image, spatial field, or other grid-valued object, use the dedicated image-generation workflow in `references/image-generation.md`.

- Default to `bf.networks.DiffusionModel(...)` with an image-capable subnet.
- Choose subnet complexity in this order: `UNet` < `UViT` < `ResidualUViT`.
- `ConvolutionalNetwork` is a **summary network for image conditions**, not the default choice for image-valued targets.
- Ensure all conditioning information is channel-wise concatenable with the image target; broadcast global conditions from `(B, D)` to `(B, H, W, D)` before concatenation.

## Training inspection

After every `fit_online`, `fit_offline`, or `fit_disk` call, you **must** perform the following steps. Do not skip any of them.

### Always pass `validation_data`

All three `fit_*` methods accept a `validation_data` argument. When a simulator is available, pass an integer (e.g., `validation_data=300`) to auto-simulate validation sets. When training offline or from disk without a simulator, pass a held-out dict of arrays. This enables `val_loss` tracking and overfitting detection.

### Always save and inspect the returned History

`fit_*` returns a `keras.callbacks.History` object. `history.history` is a dict with `"loss"` (always present) and `"val_loss"` (present when `validation_data` is provided). **BayesFlow does not save the history automatically** — you must save it yourself.

After training, always:

1. **Save the history** to a JSON file so it survives the session:
   ```python
   import json
   with open("history.json", "w") as f:
       json.dump(history.history, f)
   ```
2. **Run `scripts/inspect_training.py`** to get a structured convergence report:
   ```bash
   python scripts/inspect_training.py --history history.json
   ```
   Or call it in-process:
   ```python
   from scripts.inspect_training import inspect_history
   report = inspect_history(history.history)
   ```
3. The script checks for: NaN in losses, overfitting (val_loss ratio > 1.5×), under-training (loss still decreasing), and prints a JSON report with go/no-go recommendation.

### Controlling terminal output

All `fit_*` methods pass `**kwargs` through to Keras `model.fit()`. Use `verbose=1` (default) for progress bars, `verbose=2` for one line per epoch, or `verbose=0` to suppress output. In non-interactive environments (e.g., scripts piped to a file), prefer `verbose=2`.

## Diagnostics gate

This section applies to **low-dimensional inferential targets** such as scalar or vector parameters. For image-valued targets, do **not** use `scripts/check_diagnostics.py`; instead, save visual diagnostics and inspect a small grid of generated samples from held-out conditions.

After computing diagnostics with `workflow.compute_default_diagnostics(test_data=..., as_data_frame=True)`, you **must** perform the following steps. Do not skip any of them.

### Always save and check diagnostics

The returned DataFrame has metric names as rows (`RMSE`, `Log-gamma`, `ECE`, `Post. Contraction`) and parameter names as columns.

After computing diagnostics, always:

1. **Print the full DataFrame** so values are visible in the output.
2. **Save the DataFrame** to CSV:
   ```python
   metrics.to_csv("metrics.csv")
   ```
3. **Run `scripts/check_diagnostics.py`** to get a structured pass/fail report:
   ```bash
   python scripts/check_diagnostics.py --metrics metrics.csv
   ```
   Or call it in-process:
   ```python
   from scripts.check_diagnostics import check_diagnostics
   report = check_diagnostics(metrics)
   ```
4. The script checks each parameter against house thresholds and returns a JSON report with a `GO`, `WARN`, or `STOP` decision.

### Go / no-go decision

- If **any parameter** has `ECE > 0.10` or `NRMSE > 0.15`: **stop**. Do not proceed to real-data inference. Diagnose and fix first (see "When things go wrong" table).
- If **any parameter** has `ECE > 0.05` or `NRMSE > 0.10`: **warn** the user and proceed only if they accept the risk.
- If contraction is `< 0.80` for parameters expected to be identifiable: **warn** — the data may not be informative for those parameters, or the summary network may need more capacity.
- If contraction is `> 0.99` with `ECE > 0.05`: **stop** — the posterior is overconfident and miscalibrated.

## Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `scripts/inspect_training.py` | Check training convergence | `--history history.json` | JSON report: NaN, overfitting, under-training |
| `scripts/check_diagnostics.py` | Check diagnostics against house thresholds | `--metrics metrics.csv` | JSON report: per-parameter GO/WARN/STOP |

Both scripts can be run from the command line or imported as Python modules (see template above).

## Additional rules

- **Only online training requires a simulator.** Offline and disk workflows only require simulated pairs from the correct generative process.
- **Always keep preprocessing identical between training and inference.** Any transformation applied to simulations must also be applied to real data.
- **Do not interpret posteriors from poor diagnostics.** A sharp posterior can still be badly calibrated and vice versa.
- **Save checkpoints during training.** Neural posterior training can take time; preserve usable models and diagnostics.
- **Document the simulation budget.** Report how many simulations, which prior, and which training mode were used.
- **Report uncertainty with samples, not only point estimates.** BayesFlow gives approximate posterior draws — use them.

## Diagnostic interpretation

Use `workflow.compute_default_diagnostics(...)` as the **primary** diagnostic interface. Use `workflow.plot_default_diagnostics(...)` only as supporting visual evidence.

Prioritize these **numerical diagnostics**:

1. **Calibration**
   - Good posteriors should be well-calibrated on held-out simulations
   - Use **expected calibration error (ECE)** as the main scalar summary

2. **Recovery / estimation error**
   - Posterior summaries should ideally recover the known generating parameters across the test bank
   - Use **NRMSE** as the main scalar summary

3. **Posterior contraction**
   - The posterior should contract meaningfully relative to the prior when the data are informative
   - Use **posterior contraction** as the main uncertainty-reduction summary

Treat the following plots as **secondary diagnostics** that help explain failures, not as the primary acceptance criteria:

- **Coverage** for calibration
- **Recovery** for point-estimation behavior
- **z-score / contraction** for sensitivity and uncertainty reduction

### House thresholds

These are pragmatic workflow guardrails, not library defaults.

- **ECE**
  - `< 0.05` excellent
  - `0.05 – 0.10` acceptable
  - `> 0.10` problematic

- **Posterior contraction**
  - `< 0.80` weak information gain
  - `0.80 – 0.95` good
  - `0.95 – 0.99` excellent if calibration remains good
  - `> 0.99` inspect for possible over-concentration or train/test mismatch

- **NRMSE**
  - `< 0.05` good
  - `0.05 – 0.10` acceptable
  - `0.10 – 0.15` weak
  - `> 0.15` poor

If diagnostics disagree, trust **calibration** first. A narrow but miscalibrated posterior is worse than a wider calibrated one.

## Posterior predictive checks

PPCs in BayesFlow are always custom and model-dependent. **MUST reuse the existing simulator functions** — NEVER re-implement the generative model by hand.

General recipe:

1. Draw posterior samples `theta_s ~ q(theta | x_obs)` via `workflow.sample()`. The returned dict has the **original parameter names** (e.g., `alpha`, `beta`), each with shape `(batch, num_samples)` or `(batch, num_samples, d)`.
2. Loop over a subset of posterior draws (50 is a good default), indexing over the `num_samples` axis:
   ```python
   n_ppc = 50
   for s in range(n_ppc):
       theta_s = {k: float(samples[k][0, s]) for k in ["alpha", "beta", ...]}
       x_rep = my_observation_model(**theta_s)
       # overlay / compare x_rep to x_obs
   ```
3. Compare `x_rep` to `x_obs` using:
   - raw overlays
   - domain-relevant summary statistics
   - discrepancy measures
   - tail behavior
   - event frequencies
   - temporal or spatial structure
4. If replicated data systematically miss the observed data, improve the simulator before trusting inference

## Common gotchas

- **Using online training with a slow simulator** — switch to offline or disk training
- **Training on simulations from the wrong prior** — networks may not generalize well to real data
- **Using train simulations for diagnostics** — gives over-optimistic results
- **Ignoring preprocessing mismatch** — real-data inference breaks silently if scaling/formatting differs from training
- **Flattening structured data** — wastes inductive bias and usually hurts calibration. The most common mistake is treating N exchangeable 1D observations as a flat vector instead of turning them into a `(N, 1)` set and using a SetTransformer
- **Interpreting loss as inferential quality** — low loss does not guarantee good posterior estimation
- **Skipping PPCs** — good in-silico recovery does not guarantee the simulator explains the real data
- **Over-trusting contraction** — strong contraction without calibration can mean overconfidence
- **No checkpointing** — long training runs should always save intermediate weights

## When things go wrong

| Symptom                          | Likely Cause                                              | Fix                                                                 |
|-----------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------|
| Loss becomes NAN                       | Simulator outputs contain inf/nan or large values         | Inspect simulator outputs; if nan/inf, explore root cause; if large values, ensure everything is standardized in the workflow (e.g., divide by 255 for images) |
| Recovery good / Calibration bad   | Network is underexpressive or training is too short       | Train for twice the number of epochs; if not fixed, increase summary capacity by a factor of 2 (adjust model sizes as needed) |
| Recovery bad / Calibration good  | Some parameters are non-identifiable or same issue as bad recovery | Increase network capacity by two and train for twice as long; if no improvement, parameters may be non-identifiable |
| Nans/inf in samples on real data  | Real data preprocessed differently, contains outliers, or model mis-specified | Look at scale of real data; prompt user to test for outliers; check for potential model mis-specification |
| Online training is slow           | Batch size or simulator calls take too long               | Switch to offline training or speed up the simulator |

## Model sizes

Always check `references/model-sizes.md` for rules on model sizes when choosing a summary backbone and an inference network for a particular problem.

## Reporting template

Always report:

- inferential targets, conditions (data) and associated dimensionalities
- simulator description
- prior specification
- training mode: online / offline / disk
- simulation budget
- architecture:
  - summary network (if present)
  - inference network
- held-out diagnostic results
- posterior predictive checks (if real data present)
- limitations and likely simulator misspecifications

When the user asks for a report or mentions a non-technical audience, produce a **standalone markdown report file** explaining:

- what BayesFlow learned
- what the diagnostics say
- whether the posterior seems calibrated
- whether the model reproduces the observed data
- what remains uncertain
- suggested next steps