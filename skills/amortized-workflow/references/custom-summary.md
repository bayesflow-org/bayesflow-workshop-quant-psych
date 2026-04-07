# Custom Summary Networks

## When to use

Build a custom summary network **only** when:

1. The user **explicitly requests** a custom or specialized summary architecture, OR
2. The data modality is **non-typical** — i.e., it is NOT images, time series, sets, or plain vectors

For standard modalities, always prefer the built-in networks:

| Modality | Built-in network |
|----------|-----------------|
| Sets / exchangeable data | `SetTransformer` |
| Time series | `TimeSeriesTransformer`, `TimeSeriesNetwork`, `FusionTransformer` |
| Images | `ConvolutionalNetwork` |
| Fixed-length vectors | No summary network needed |

If the data fits one of these categories, do **not** build a custom network.

---

## How to build a custom summary network

Every custom summary network must:

1. **Inherit from `bf.networks.SummaryNetwork`**
2. **Use the `@bf.utils.serialization.serializable()` decorator** so the network can be saved and loaded
3. **Implement `call(self, x, **kwargs)`** returning a fixed-size summary tensor of shape `(batch_size, summary_dim)`
4. **Use Keras layers** — any `keras.layers.*` layer is valid inside the network

### Minimal example

```python
import keras
import bayesflow as bf

@bf.utils.serialization.serializable("custom")
class GRUSummary(bf.networks.SummaryNetwork):
    def __init__(self, summary_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.gru = keras.layers.GRU(64)
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series, **kwargs):
        """Compresses time_series of shape (batch_size, T, 1) into (batch_size, summary_dim)."""
        summary = self.gru(time_series, **kwargs)
        summary = self.summary_stats(summary)
        return summary
```

## Serialization

The `@serializable(name)` decorator registers the class so BayesFlow can save and reload it. The `name` argument is a unique string identifier — use a project-scoped name (e.g., `"my_project.networks"`) to avoid collisions.

If the network has constructor arguments beyond `**kwargs` and Python primitives, override `get_config()` and merge with `serialize()` so that `from_config` can reconstruct the object:

```python
def get_config(self):
    base_config = super().get_config()
    config = {"summary_dim": self.summary_dim, "my_param": self.my_param}
    return base_config | bf.utils.serialization.serialize(config)
```

---

## Integration with BasicWorkflow

A custom summary network plugs in the same way as any built-in one:

```python
workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=bf.networks.FlowMatching(),
    summary_network=GRUSummary(summary_dim=16),
    adapter=adapter,
)
```

Route the structured data through `summary_variables` in the adapter — the same rule as for built-in summary networks. The custom network replaces only the encoder; all other workflow steps (training, diagnostics, sampling) remain unchanged.

---
