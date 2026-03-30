# NEW_METHOD_BRIEF

## 1. Core idea

* Replace fixed text pooling in dual-encoder retrieval with **image-conditioned token pooling**.
* Introduce a **shared image-side prototype bank** as latent visual memory.
* Image embedding routes over prototypes to form a **query vector (q)**.
* This query reweights text tokens, producing a **visually grounded text representation**.
* No full cross-attention; interaction is lightweight and one-way (image → text).

---

## 2. High-level pipeline

image -> CLIP vision encoder -> V
text -> CLIP text encoder -> H

Theta_v -> normalize -> Theta_hat
Theta_hat -> self-similarity -> S_p
S_p -> row-softmax -> P_p
P_p + Theta_v -> residual aggregation -> Theta_tilde

V + Theta_tilde -> cosine similarity -> R
R -> softmax (tau_p) -> alpha

alpha + Theta_tilde -> weighted sum -> Q

Q + H -> cosine similarity -> S_t
S_t + mask -> softmax (tau_t) -> beta

beta + H -> weighted sum -> T_pool

V -> projector -> Z_v
T_pool -> projector -> Z_t

Z_v, Z_t -> normalize -> cosine logits
logits -> symmetric InfoNCE

Theta_v -> diversity loss

---

## 3. What is new vs baseline

* Global **prototype bank (Theta_v)** shared across dataset
* **Parameter-free prototype self-interaction** (contextualization)
* **Image-to-prototype routing (alpha)**
* **Prototype-based query vector (Q)**
* **Query-guided token weighting (beta)** instead of fixed pooling
* **Prototype diversity regularization**

---

## 4. Fixed design decisions (v1)

* Backbone: pretrained CLIP ViT-B/16 + paired CLIP text encoder 
* Freeze policy: frozen backbones (initially) 
* Prototype bank: single shared, N = 32, dim = D 
* Prototype init: random (default), normalized; k-means optional 
* Contextualization: normalized self-interaction + residual 
* Routing similarity: cosine
* Token scoring similarity: cosine
* Token pooling: content tokens only (exclude special tokens) 
* Summary vector: single Q (no multi-head)
* Projector: symmetric 2-layer MLP, output dim = 256 
* Loss: InfoNCE + diversity (no balancing loss) 
* Output normalization: L2 normalize before InfoNCE

---

## 5. Not included in v1

* no cross-attention between image and text
* no patch-level or token-level image features
* no multi-head or multi-query summaries
* no sparse/top-k routing (softmax only)
* no learnable attention scorer (no MLP scorer)
* no balancing loss (MoE-style)
* no entropy regularization on alpha or beta
* no multiple prototype banks or hierarchical memory
* no backbone full finetuning

---

## 6. Key implementation implication

* Replace **text pooling path**: remove EOS/CLS pooling → add query-guided pooling (Q → beta → T_pool).
* Add new modules:

  * prototype bank parameter (Theta_v)
  * prototype contextualization block
  * routing (alpha) and token weighting (beta)
* Keep reusable:

  * CLIP encoders
  * contrastive training loop
  * projection heads (slightly extended)
* Ensure strict masking + normalization + correct softmax axes (critical for correctness).
