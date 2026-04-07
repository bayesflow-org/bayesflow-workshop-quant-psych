# BayesFlow Conditioning Logic

For amortized posterior inference, BayesFlow trains an estimator approximating:

$$p(\text{inference\_variables} \mid f(\text{summary\_conditions}),\; \text{inference\_conditions})$$

where $f$ is an optional summary network with a proper inductive bias. Understanding which data goes where is essential for setting up `BasicWorkflow` correctly.

In practice, data is routed to the right slots via a `bf.Adapter()` pipeline. See `adapter.md` for the full adapter API — including examples of `.as_set()`, `.constrain()`, `.broadcast()`, and `.concatenate()` — and for the distinction between using an explicit adapter vs. the `BasicWorkflow` naming shorthand.

---

## The two conditioning slots

### `inference_conditions`

Data passed **directly and unchanged** to the inference network. No summary network is applied.

**Use this for:**
- Scalars or fixed-length feature vectors (e.g., a single design matrix row, a scalar covariate, a precomputed statistic)
- Data that is already a **single vector** with no structural dimensions to remove
- Any quantity whose dimensionality is the same for every simulation (i.e., its shape does not vary across the batch)

```python
workflow = bf.BasicWorkflow(
    ...
    inference_conditions=["scalar_covariate", "fixed_feature_vector"],
    # these are concatenated with the target variables before entering the inference network
)
```

### `summary_variables`

Data passed **first through a summary network** before being concatenated with the target variables.

**Use this for:**
- **Set-based / exchangeable data** — N i.i.d. observations, regression datasets of (x, y) pairs, repeated measurements. Shape: `(batch, N, d)` → compressed to `(batch, summary_dim)` by a `SetTransformer`.
- **Time series** — ordered sequences. Shape: `(batch, T, d)` → compressed by `TimeSeriesTransformer` or `TimeSeriesNetwork`.
- **Images** — grid data. Shape: `(batch, H, W, C)` → compressed by `ConvolutionalNetwork`.
- Any data with **structural dimensions** (axes beyond the batch axis) that need to be collapsed.

```python
workflow = bf.BasicWorkflow(
    ...
    summary_network=bf.networks.SetTransformer(...),
    summary_variables=["observations"],  # (batch, N, d) → compressed to (batch, summary_dim)
)
```

---

## A workflow can use both simultaneously

`inference_conditions` and `summary_variables` are not mutually exclusive. A common pattern is:

- **`summary_variables`**: the structured, variable-size observations (e.g., N data points)
- **`inference_conditions`**: fixed scalar metadata known at inference time (e.g., sample size N, experimental design, scalar context variables)

```python
workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=bf.networks.FlowMatching(),
    summary_network=bf.networks.SetTransformer(...),
    inference_variables=["parameters"],
    summary_variables=["observations"],        # (batch, N, d) → summary network → (batch, summary_dim)
    inference_conditions=["sample_size"],       # (batch, 1) → concatenated directly
)
```

At inference time the network sees:
```
[target_vars | summary_net(observations) | sample_size]
```

---

## Decision table

| Data type | Shape example | Slot | Network |
|---|---|---|---|
| Scalar covariate | `(batch, 1)` | `inference_conditions` | none |
| Fixed feature vector | `(batch, d)` | `inference_conditions` | none |
| N i.i.d. observations | `(batch, N, d)` | `summary_variables` | `SetTransformer` |
| Regression dataset (x, y pairs) | `(batch, N, 2)` | `summary_variables` | `SetTransformer` |
| Time series | `(batch, T, d)` | `summary_variables` | `TimeSeriesTransformer` |
| Image | `(batch, H, W, C)` | `summary_variables` | `ConvolutionalNetwork` |
| Precomputed summary statistic | `(batch, s)` | `inference_conditions` | none |

---

## Common mistakes

- **Putting set-based data in `inference_conditions`** — this forces the agent to flatten `(N, d)` into a single vector, destroying permutation-invariance and degrading inference. Always route exchangeable data through `summary_variables` + `SetTransformer`.
- **Using `inference_conditions` when N varies across simulations** — the inference network expects fixed-length inputs; variable-N data must go through a summary network.
- **Omitting `summary_variables` key from simulator output** — the key name in the simulator's return dict must match exactly what is listed in `summary_variables`.
