# Config Path Flag Reference (New Prototype Flags)

This reference is path-first (YAML config path -> CLI flag).
It documents the newly introduced prototype-routing and weighted-retrieval flags.

## Scope

- All `configs/**/*.yaml` files with a `prototype:` block now include the local-routing keys.
- All `configs/**/*.yaml` files with an `objectives:` block already include the weighted-retrieval keys.

## Prototype Routing (Path-First)

| YAML config path | CLI flag | Description | Allowed YAML values | Default in configs |
|---|---|---|---|---|
| `prototype.routing_source` | `--prototype_routing_source` | Selects routing input source for prototype assignment. | `global` \| `local_evidence` | `global` |
| `prototype.local_routing_temperature` | `--prototype_local_routing_temperature` | Temperature for local token -> prototype evidence logits. | positive float (e.g., `0.03`, `0.07`, `0.1`) | `0.07` |
| `prototype.local_routing_pooling` | `--prototype_local_routing_pooling` | Pooling across local tokens before softmax over prototypes. | `logsumexp` \| `max` \| `mean` (aliases accepted by code: `lse`, `avg`) | `logsumexp` |
| `prototype.local_routing_use_adapter` | `--prototype_local_routing_use_adapter` | Enables lightweight local-token adapter before routing. | `true` \| `false` | `true` |
| `prototype.local_routing_adapter_dim` | `--prototype_local_routing_adapter_dim` | Hidden width for local routing adapter MLP. | `null` (empty) or positive int (e.g., `128`, `256`, `512`) | `null` |
| `prototype.local_routing_normalize_inputs` | `--prototype_local_routing_normalize_inputs` | L2-normalize local tokens/prototypes before similarity. | `true` \| `false` | `true` |

## Weighted Retrieval Loss (Path-First)

| YAML config path | CLI flag | Description | Allowed YAML values | Typical/default |
|---|---|---|---|---|
| `objectives.objectives.use_loss_weight_ret` | `--use_loss_weight_ret` | Enables weighted prototype retrieval loss term. | `true` \| `false` | `false` |
| `objectives.objectives.weight_ret_margin_delta` | `--weight_ret_margin_delta` | Margin threshold used in host-difficulty weighting. | float (e.g., `0.0`, `0.1`, `0.2`) | `0.0` |
| `objectives.objectives.weight_ret_detach_host` | `--weight_ret_detach_host` | Detaches host logits when forming per-sample weights. | `true` \| `false` | `true` |
| `objectives.objectives.weight_ret_normalize_mean_one` | `--weight_ret_normalize_mean_one` | Normalizes weights to mean=1 for scale stability. | `true` \| `false` | `true` |
| `objectives.lambda.weight_ret` | `--lambda_weight_ret` | Multiplier for `loss_weight_ret`. | float `>= 0` (e.g., `0.0`, `0.1`, `0.5`, `1.0`) | `0.0` |

## Important Note About `weight_ret_tau`

- Runtime CLI flag still exists: `--weight_ret_tau`
- Parser target variable: `weight_ret_tau`
- Current policy in configs (per your request): do not add `objectives.objectives.weight_ret_tau` in YAML; rely on default/runtime CLI override when needed.

Recommended usage when you want to override at runtime:

```bash
python train.py --config_file configs/itself_prototype_from_legacy.yaml --weight_ret_tau 0.5
```

## Minimal YAML Example

```yaml
prototype:
  routing_source: local_evidence
  local_routing_temperature: 0.07
  local_routing_pooling: logsumexp
  local_routing_use_adapter: true
  local_routing_adapter_dim:
  local_routing_normalize_inputs: true

objectives:
  objectives:
    use_loss_weight_ret: true
    weight_ret_margin_delta: 0.0
    weight_ret_detach_host: true
    weight_ret_normalize_mean_one: true
  lambda:
    weight_ret: 0.2
```
