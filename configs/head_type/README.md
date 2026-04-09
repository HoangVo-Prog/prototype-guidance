# Head-Wise Configs

This folder stores host/head-type specific config variants.

## Structure

- `configs/head_type/itself/*.yaml`
  - `host.type: itself`
  - backbone optimizer defaults:
    - `optimizer.lr_image_backbone: 5.0e-6`
    - `optimizer.lr_text_backbone: 5.0e-6`
  - schedule entries with `lr_overrides.host_backbone` are set to `5.0e-6`

- `configs/head_type/clip/*.yaml`
  - `host.type: clip`
  - backbone optimizer defaults:
    - `optimizer.lr_image_backbone: 1.0e-5`
    - `optimizer.lr_text_backbone: 1.0e-5`
  - schedule entries with `lr_overrides.host_backbone` are set to `1.0e-5`

Each file keeps the same direction/recipe semantics as its source config, but with host/head-specific defaults.
