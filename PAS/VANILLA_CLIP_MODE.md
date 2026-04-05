# Vanilla CLIP Mode

`model.training_mode: vanilla_clip` adds a strict CLIP-style retrieval baseline to this repo.

What this mode does:
- disables the prototype bank, routing, contextualization, and basis-bank construction
- disables image-conditioned pooling
- uses the CLIP EOS token as the text representation
- trains with symmetric CLIP retrieval loss: image-to-text plus text-to-image
- evaluates with plain global image-text similarity

What this mode does not do:
- it does not reuse the existing direct no-prototype head with learned token pooling
- it does not enable proxy, diagonal fidelity, support, balancing, or diversity losses

Projector behavior:
- `model.use_custom_projector: false` means strict vanilla CLIP embeddings from the CLIP backbone are used directly
- `model.use_custom_projector: true` keeps the CLIP retrieval formulation but adds the repo's custom MLP projectors on top of CLIP global embeddings

Recommended baseline config:
- `PAS/configs/baselines/vanilla_clip.yaml`

Required vanilla settings:
- `model.training_mode: vanilla_clip`
- `model.use_prototype_bank: false`
- `model.use_image_conditioned_pooling: false`
- `text_pooling.token_policy: eos_only`
- `loss.retrieval_mode: clip_bidirectional`
- `evaluation.retrieval_scorer: exact`

Repo limitation:
- this mode still uses the repo's current CLIP wrapper and therefore inherits its supported backbone set and tokenization behavior
- within that wrapper, the text embedding is the EOS token from `encode_text_intermediates(...)`, which is the strict CLIP-style text representation used here
