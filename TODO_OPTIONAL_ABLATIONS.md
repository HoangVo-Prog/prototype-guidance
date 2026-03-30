# TODO_OPTIONAL_ABLATIONS

Only ablation-valid or optional-later items are listed here. Core logic conflicts were resolved in code and are not deferred.

- Sweep `num_prototypes` over `{16, 32, 64}`.
- Disable prototype contextualization as a no-context ablation.
- Switch contextualization mode from `residual` to `overwrite` as an ablation.
- Compare routing similarity `cosine` vs `dot`.
- Compare token scoring similarity `cosine` vs `dot`.
- Enable balancing loss only if prototype underuse is observed.
- Compare `normalized_random` vs sampled-image vs k-means prototype initialization.
- Run `content_plus_special` and `eos_only` token-policy ablations.
- Add baseline poolers such as mean pooling or EOS-only outside the default path.
- Compare `mlp2` vs linear projector heads.
- Sweep `tau_p` and `tau_t` around the fixed default `0.07`.
- Partially unfreeze the last backbone block(s) after the frozen-backbone default is verified.
