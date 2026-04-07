# BayesFlow Adapters

An `Adapter` is a composable preprocessing pipeline that transforms the raw simulator output dict into the exact tensor format expected by the networks. It runs identically during training and inference — ensuring preprocessing consistency automatically.

---

## When to use a custom adapter vs. the workflow shorthand

> **CRITICAL: The explicit `adapter=` and the naming shorthand (`inference_variables=`, `summary_variables=`, etc.) are MUTUALLY EXCLUSIVE. NEVER pass both — they will silently conflict. When in doubt, always use an explicit `bf.Adapter()` chain.**

`BasicWorkflow` accepts `inference_variables`, `inference_conditions`, and `summary_variables` as keyword arguments. When you pass these names, the workflow constructs a **minimal implicit adapter** that only renames and routes the named keys. This is convenient for simple cases but cannot handle:

- structural transformations (e.g., stacking x and y into a set)
- parameter-space constraints (e.g., `sigma > 0`)
- dtype conversion
- broadcasting scalars to the batch dimension
- any derived quantity not already present in the simulator output

For any of these, pass an explicit `adapter=` to `BasicWorkflow`. The explicit adapter replaces the implicit one entirely. **Do NOT write a custom Python function as a substitute for `bf.Adapter()`** — the adapter handles both training and inference preprocessing identically, which a manual function cannot guarantee.

```python
workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=inference_net,
    summary_network=summary_net,
    adapter=adapter,            # explicit adapter — replaces the naming shorthand
    checkpoint_filepath="checkpoints",
    checkpoint_name="my_model",
)
```

---

## Minimal working example — linear regression

The simulator defines a prior over regression parameters and a likelihood that generates N observations. Note that `N` is passed as an argument to `likelihood` by `bf.make_simulator`, so it appears in the simulator output dict alongside `beta`, `sigma`, `x`, and `y`. Varying `N` between (as opposed to within) batches qualifies it as a "meta function".

```python
import numpy as np
import bayesflow as bf

def prior():
    beta = np.random.normal([2, 0], [3, 1])
    sigma = np.random.gamma(1, 1)
    return dict(beta=beta, sigma=sigma)

def likelihood(beta, sigma, N):
    x = np.random.normal(0, 1, size=N)
    y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
    return dict(y=y, x=x)

def meta():
    return dict(N=np.random.randint(5, 50))

simulator = bf.make_simulator([prior, likelihood], meta_fn=meta)
```

The adapter then transforms the raw output dict `{beta, sigma, N, x, y}` into the tensor layout the networks expect:

```python
adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x", "y"]) # use .as_time_series() for time series
    .constrain("sigma", lower=0)
    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)
```

---

## Step-by-step explanation

### `.broadcast("N", to="x")`

The simulator returns `N` as a scalar (or 0-d array) — a single value shared across all datasets in a batch. Networks, however, expect every tensor to have a leading batch dimension. `.broadcast("N", to="x")` replicates `N` along the batch dimension, inferring the required batch size from `x`.

Use this whenever the simulator outputs a scalar context variable (e.g., sample size, design parameter, temperature) that must later be concatenated with per-dataset tensors.

For **conditional image generation**, apply the same principle spatially: if a condition starts as `(B, D)` but must be concatenated channel-wise with an image target `(B, H, W, C)`, broadcast or tile it to `(B, H, W, D)` in the adapter or simulator before concatenation.

### `.as_set(["x", "y"])`

Marks `x` and `y` as exchangeable — converting them into matrices of shape `(N, 1)`. This is only needed for 1D sequences missing a trailing axis.

**This must be used whenever the observations are i.i.d. given the parameters.** Omitting it while still using a `SetTransformer` will fail: all bayesflow summary networks expect at least 3D tensors of shape `(batch_size, ..., D)`.

### `.constrain("sigma", lower=0)`

Maps `sigma` through a bijection (by default: softplus) so that the neural network that predicts in unconstrained space is guaranteed to produce valid (positive) values when back-transformed. This is equivalent to a log or softplus reparameterisation and is required for any parameter with a hard boundary:

| Constraint | Typical use |
|---|---|
| `lower=0` | standard deviations, rates, variances |
| `lower=0, upper=1` | probabilities, mixing weights |
| `lower=a, upper=b` | bounded parameters |

Without this, the network may predict values outside the support, causing NaN posteriors or nonsensical samples.

### `.sqrt("N")`

Applies `sqrt` to `N` before it enters the inference network. This is a feature-engineering step — `sqrt(N)` grows more smoothly than `N` and is a better signal for the network about observation scale. Apply similar monotone transforms to any context variable with a wide dynamic range.

### `.convert_dtype("float64", "float32")`

Converts all `float64` tensors to `float32`. Neural network backends (JAX, PyTorch) default to `float32`; passing `float64` causes silent dtype promotion or errors. Always call this when the simulator produces `float64` arrays (NumPy's default).

### `.concatenate(["beta", "sigma"], into="inference_variables")`

Stacks `beta` (a vector) and `sigma` (a scalar) along the last axis into a single tensor named `"inference_variables"`. The inference network always expects its targets as a single concatenated array. Use `.concatenate(...)` to assemble all parameters — the order here defines the column order in posterior samples.

### `.concatenate(["x", "y"], into="summary_variables")`

Stacks `x` and `y` along the feature axis into a single `(batch, N, 2)` tensor named `"summary_variables"`, which is then routed to the summary network. The name `"summary_variables"` is BayesFlow's reserved key for summary network input.

### `.rename("N", "inference_conditions")`

Renames the preprocessed `N` tensor to `"inference_conditions"` — BayesFlow's reserved key for data passed directly to the inference network without a summary network. `.rename` is used for any final routing step that does not change values, only keys.

---

## Reserved key names

BayesFlow recognises these names as routing targets:

| Key | Destination |
|---|---|
| `"inference_variables"` | inference network target (what is being inferred) |
| `"inference_conditions"` | inference network input, passed directly |
| `"summary_variables"` | summary network input |

Any key not renamed to one of these is ignored during training and inference.

---

## Order of adapter steps matters

Steps execute sequentially. Common ordering conventions:

1. **Structural transforms first** — `.broadcast(...)`, `.as_set(...)`
2. **Parameter-space constraints** — `.constrain(...)`
3. **Feature engineering / scaling** — `.sqrt(...)`, `.log(...)`, `.standardize(...)`
4. **Dtype conversion** — `.convert_dtype(...)` (always near the end)
5. **Concatenation and renaming** — `.concatenate(...)`, `.rename(...)` (always last)

---

## Adapter pitfalls

- **Forgetting `.convert_dtype`** — NumPy defaults to `float64`; most backends default to `float32`. This mismatch causes runtime errors or silent type promotion.
- **Concatenating before constraining** — if you concatenate parameters and then constrain, the constraint applies to the concatenated block, not individual columns. Always constrain before concatenating.
- **Omitting `.broadcast` for scalar context variables (shared within batch)** — scalars passed as `inference_conditions` without broadcasting will fail or be silently dropped during batch assembly.
- **Failing to spatially broadcast image conditions** — image diffusion backbones concatenate conditions channel-wise with the target image. Global conditions such as `(B, D)` must be expanded to `(B, H, W, D)` before concatenation.
- **Using the naming shorthand with a custom adapter simultaneously** — if you pass `adapter=`, do not also pass `inference_variables=`, `summary_variables=`, etc. to `BasicWorkflow`; they will conflict.
- **No need to call the adapter manually** - the adapter is part of an `approximator` and will always be called internally in the context of the appropriate call (e.g., forward, inverse).