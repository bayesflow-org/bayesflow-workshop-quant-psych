# Image Generation and Spatial Outputs

Use this reference when the **inferential target is an image or spatial field**, not a low-dimensional parameter vector.

Typical examples:
- conditional image generation
- Bayesian denoising
- Gaussian random field creation
- spatial emulator outputs
- image-valued posterior or likelihood targets

This is a different workflow from the default data-to-parameters case.

## When to use this workflow

Use the image-generation workflow when:
- the object being sampled is an image-like tensor, e.g. `(H, W, C)`
- the task is to generate or reconstruct spatial outputs conditioned on metadata, noisy inputs, or covariates
- the target has medium-to-high dimensional spatial structure and should not be flattened

Do **not** use this workflow when:
- the image is only an observed condition and the inferential target is still low-dimensional
- you only need an image summary before predicting a small parameter vector

In that case, keep the default workflow and use `bf.networks.ConvolutionalNetwork` as a summary network.

## Core rule

For image-valued targets, default to a diffusion / flow matching inference network with an image-capable subnet:

```python
import bayesflow as bf

diffusion = bf.networks.DiffusionModel(
    subnet=bf.networks.UNet,
    prediction_type="velocity",
    noise_schedule="cosine"
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=diffusion,
    adapter=adapter,
    initial_learning_rate=1e-4
)
```

This is the correct default for conditional image generation.

## Subnet choice

Choose image backbones in increasing complexity:

1. `bf.networks.UNet`
2. `bf.networks.UViT`
3. `bf.networks.ResidualUViT`

Use them as follows:
- **`UNet`**: default starting point; use first for denoising, smaller images, or simpler spatial structure
- **`UViT`**: use when `UNet` underfits or long-range spatial interactions matter more
- **`ResidualUViT`**: use for the hardest image-generation tasks or when diagnostics still fail after sufficient training with `UViT`

Escalate capacity only after training has converged and held-out diagnostics still show weak recovery or poor calibration.

## Conditioning shape requirements

Image diffusion backbones expect the condition to be **channel-wise concatenable** with the image target.

If the image target has shape:
- `x.shape == (B, H, W, C)`

then the condition passed into the subnet must already have compatible spatial shape:
- `cond.shape == (B, H, W, D)`

A global condition such as `(B, D)` is **not** enough by itself. Broadcast or tile it first, either:
- in the simulator, or
- in the adapter

before the workflow reaches the diffusion subnet.

## Conditioning patterns

### Global metadata to spatial map

Example: scalar parameters, experimental settings, or latent variables stored as `(B, D)`.

These must be expanded to `(B, H, W, D)` before concatenation with the target image.

Conceptually:

```python
# parameters: (B, D)
# target image: (B, H, W, C)
# desired condition map: (B, H, W, D)
```

### Image-to-image conditioning

Example: noisy image to clean image, partial field to completed field, masked image to reconstructed image.

If the conditioning object is already image-shaped `(B, H, W, Dc)`, it can usually be concatenated directly as channels after dtype/scale alignment.

### Mixed conditioning

You can combine both:
- image-shaped condition maps
- broadcast metadata maps

Concatenate them into a single condition tensor before passing them to the inference network.

## Minimal workflow sketch

```python
import bayesflow as bf

simulator = bf.make_simulator([prior, observation_model])

adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    # image target -> inference_variables
    .rename("target_image", "inference_variables")
    # condition map must already be broadcast to (B, H, W, D)
    .rename("condition_map", "inference_conditions")
)

diffusion = bf.networks.DiffusionModel(
    subnet=bf.networks.UNet,
    prediction_type="velocity",
    noise_schedule="cosine"
)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    inference_network=diffusion,
    adapter=adapter,
    initial_learning_rate=1e-4
)
```

## Practical guidance

- Start with `UNet`.
- Keep image preprocessing identical between simulation, training, and inference.
- Standardize or scale image intensities consistently if needed.
- Verify that condition maps and target images have matching spatial dimensions before training.
- Use held-out simulations for diagnostics, but do **not** run `scripts/check_diagnostics.py` here; that script is designed for low-dimensional parameter targets.
- For now, the default image-generation check is to plot a small grid of generated samples from a few held-out conditions.

## Minimal visual check

For image-valued targets, the current house rule is simple: draw a few samples from held-out conditions and inspect them visually in a grid.

```python
held_out = workflow.simulate(5)

samples = workflow.sample(
    conditions={"condition_map": held_out["condition_map"]},
    num_samples=4,
)

# sample key names follow the adapter reverse transform
generated = samples["target_image"]  # e.g. (5, 4, H, W, C)

# Plot a small grid for quick inspection.
# For now, prefer this over check_diagnostics() for image-valued targets.
```

## Common mistakes

- Using `ConvolutionalNetwork` when the target itself is an image
- Leaving global conditions at shape `(B, D)` and expecting the subnet to broadcast them implicitly
- Running `scripts/check_diagnostics.py` on image-valued outputs
- Mixing image-generation guidance with the standard low-dimensional parameter workflow
- Re-implementing the target simulator for PPCs instead of reusing the existing forward model

## Example

BayesFlow example notebook:
- https://github.com/bayesflow-org/bayesflow/blob/main/examples/Spatial_Data_and_Parameters.ipynb

Use this as a concrete reference for spatial/image data workflows.
