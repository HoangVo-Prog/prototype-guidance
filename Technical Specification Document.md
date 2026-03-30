### Prototype-conditioned Text Pooling via an Image-side Visual Memory Bank

This document is a **paper-ready technical specification** for implementation, debugging, experimentation, and method writeup. It is written as an internal companion to the eventual Method and Appendix sections. The goal is to remove hidden assumptions, fix defaults for a first reproducible implementation, and clearly separate **core design choices** from **ablation variables**.

---

# 1. Scope and claim boundary

## 1.1 Objective

The method is designed to improve image-text interaction so that the final text representation is **more tightly grounded in visual evidence** without relying on full pairwise cross-attention between image and text tokens.

The central mechanism is an **image-side prototype bank** that serves as a shared latent visual memory:

1. it stores reusable visual anchors across the dataset,
    
2. it produces an image-conditioned summary vector for each sample,
    
3. that summary vector reweights token-level text features,
    
4. yielding a pooled text representation that is more responsive to the actual image evidence.
    

## 1.2 Claim boundary for paper writing

The method should be described as learning a **latent reusable visual memory** or **shared visual anchors**. It should **not** overclaim that each prototype corresponds to a human-interpretable semantic concept, object, part, or attribute.

Safe phrasing:

- shared visual anchors
    
- latent visual modes
    
- image-side memory bank
    
- reusable visual basis for text grounding
    

Avoid:

- each prototype represents a semantic concept
    
- prototypes explicitly correspond to discrete visual entities
    
- interpretable concept discovery, unless proven empirically
    

---

# 2. Notation and tensor conventions

Let:

- (B): batch size
    
- (L): padded text length
    
- (D): shared backbone feature dimension
    
- (N): number of visual prototypes
    
- (d_o): projector output dimension
    

Inputs:

- image batch: (X \in \mathbb{R}^{B \times C \times H \times W})
    
- tokenized text: (T \in \mathbb{N}^{B \times L})
    
- attention mask: (M \in {0,1}^{B \times L})
    

Backbone outputs:

- global image embedding:  
$$
    V = f_v(X) \in \mathbb{R}^{B \times D}  
$$
- token-level text hidden states:  
$$
    H = f_t(T, M) \in \mathbb{R}^{B \times L \times D}  
$$
Prototype bank:  
$$
\Theta_v =  
\begin{bmatrix}  
\theta_1 \\  
\theta_2 \\  
\vdots \\  
\theta_N  
\end{bmatrix}  
\in \mathbb{R}^{N \times D}  
$$
All tensors are batch-first. All softmax dimensions must be explicitly specified in code.

---

# 3. Method overview

For sample (i), the method proceeds as follows:

1. obtain a global image embedding (v_i),
    
2. obtain token-level text features (H_i = [h_{i,1}, \dots, h_{i,L_i}]),
    
3. contextualize a global prototype bank (\Theta_v) into (\tilde{\Theta}_v),
    
4. route (v_i) over (\tilde{\Theta}_v) to obtain routing weights (\alpha_i),
    
5. form an image-conditioned prototype summary (q_i),
    
6. use (q_i) to score text tokens,
    
7. compute token weights (\beta_i),
    
8. pool the text tokens into (t_i^{\text{pool}}),
    
9. project image and pooled text representations,
    
10. optimize them with symmetric InfoNCE.
    

This yields a lightweight structured interaction module in which the image influences the final text representation **through a reusable latent bank**, rather than through full cross-attention.

---

# 4. Architecture specification

## 4.1 Backbones

### Visual encoder
$$
V = f_v(X) \in \mathbb{R}^{B \times D}  
$$
Default choice:

- use the pretrained **global image embedding** from a CLIP-style visual encoder.
    

This should be a single vector per image, not a set of patch tokens, for the first implementation.

### Text encoder
$$
H = f_t(T, M) \in \mathbb{R}^{B \times L \times D}  
$$
Default choice:

- use the **last-layer token hidden states** from the paired text encoder.
    

These are token-level representations, not a pre-pooled CLS/EOS output.

---

## 4.2 Prototype bank

The prototype bank is a **global trainable parameter** shared across all samples and all batches:  
$$
\Theta_v \in \mathbb{R}^{N \times D}  
$$
Default:

- a single shared bank
    
- trainable end to end
    
- no sample-specific bank
    
- no per-head bank in v1
    
- prototype dimension equals the backbone feature dimension (D)
    

Gradient flow:

- gradients from routing, text pooling, and contrastive loss all propagate into (\Theta_v)
    
- if prototype regularization is used, it also directly updates (\Theta_v)
    

---

## 4.3 Prototype contextualization

The current idea uses parameter-free self-attention over prototypes:  
$$
\tilde{\Theta}_v =
\operatorname{softmax}  
\left(  
\frac{\Theta_v \Theta_v^\top}{\sqrt D}  
\right)\Theta_v  
$$
### Implementation form

Let:  
$$
\hat{\Theta}_v = \operatorname{norm}(\Theta_v)  
$$
where row-wise L2 normalization is applied for similarity computation.

Then:  
$$
S_p = \hat{\Theta}_v \hat{\Theta}_v^\top \in \mathbb{R}^{N \times N}  
$$
$$
A_p = \frac{S_p}{\sqrt D}  
$$
$$
P_p = \operatorname{softmax}(A_p, \text{dim}=-1) \in \mathbb{R}^{N \times N}  
$$
Recommended default:  
$$
\tilde{\Theta}_v = \Theta_v + P_p \Theta_v  
$$
This residual form is preferred over a pure overwrite for v1 because it preserves prototype identity while still enabling prototype interaction.

### Softmax axis

- row-wise over the prototype dimension
    
- each row of (P_p) sums to 1
    

### Operation types

- (\Theta_v \Theta_v^\top): matrix multiplication
    
- softmax: row-wise normalization over prototypes
    
- (P_p \Theta_v): weighted sum over prototypes
    

### Gradient flow

- gradients flow through both (\Theta_v) terms in the residual version
    
- no detach is recommended in v1
    

---

## 4.4 Image-conditioned routing

Given image embedding (V \in \mathbb{R}^{B \times D}) and contextualized prototypes (\tilde{\Theta}_v \in \mathbb{R}^{N \times D}), routing logits are:  
$$
R = \operatorname{sim}(V, \tilde{\Theta}_v) \in \mathbb{R}^{B \times N}  
$$
Recommended default:  
$$
R = \hat{V} \hat{\tilde{\Theta}}_v^\top  
$$
where both (V) and (\tilde{\Theta}_v) are row-wise L2 normalized for the similarity computation.

Routing weights:  
$$
\alpha = \operatorname{softmax}\left(\frac{R}{\tau_p}, \text{dim}=-1\right) \in \mathbb{R}^{B \times N}  
$$
For sample (i):  
$$
\alpha_{i,n} =
\frac{\exp(\operatorname{sim}(v_i,\tilde{\theta}_n)/\tau_p)}  
{\sum_{m=1}^N \exp(\operatorname{sim}(v_i,\tilde{\theta}_m)/\tau_p)}  
$$
### Softmax axis

- over the prototype dimension (N)
    

### Default

- dense soft routing
    
- no sparsemax, entmax, or top-k in v1
    

---

## 4.5 Image-conditioned prototype summary

The image-conditioned summary is:  
$$
Q = \alpha \tilde{\Theta}_v \in \mathbb{R}^{B \times D}  
$$
For sample (i):  
$$
q_i = \sum_{n=1}^N \alpha_{i,n} \tilde{\theta}_n  
$$
This is a single summary vector per sample.

### Default

- single-vector summary only in v1
    
- no multi-head extension in the first implementation
    

---

## 4.6 Query-guided token scoring

Given (Q \in \mathbb{R}^{B \times D}) and (H \in \mathbb{R}^{B \times L \times D}), compute token scores:  
$$
S_t \in \mathbb{R}^{B \times L}  
$$
For each sample (i) and token (j):  
$$
s_{i,j} = \frac{\operatorname{sim}(q_i, h_{i,j})}{\tau_t}  
$$
Recommended default:  
$$
s_{i,j} =
\frac{\hat{q}_i^\top \hat{h}_{i,j}}{\tau_t}  
$$
### Token masking

Before softmax:

- padded tokens must be assigned (-\infty)
    
- excluded special tokens must also be assigned (-\infty)
    

---

## 4.7 Token weights
$$
\beta = \operatorname{masked_softmax}(S_t, \text{dim}=-1) \in \mathbb{R}^{B \times L}  
$$
For sample (i):  
$$
\beta_{i,j} =
\frac{\exp(s_{i,j})}  
{\sum_{k \in \mathcal{V}_i} \exp(s_{i,k})}  
$$
where (\mathcal{V}_i) is the set of valid tokens after padding and special-token masking.

### Softmax axis

- over the token dimension (L)
    

---

## 4.8 Pooled text representation
$$
T^{\text{pool}} \in \mathbb{R}^{B \times D}  
$$
with:  
$$
t_i^{\text{pool}} = \sum_{j=1}^{L} \beta_{i,j} h_{i,j}  
$$
Batch implementation:  
$$
T^{\text{pool}} = \beta H  
$$
where the multiplication is a weighted sum over tokens.

### Default token inclusion policy

For the main model:

- exclude padding tokens
    
- exclude special tokens by default
    

Implement as ablations:

- include all tokens
    
- include content tokens only
    
- EOS-only baseline
    
- mean-pooling baseline
    
- CLS-only baseline if the backbone supports a standard CLS usage
    

---

## 4.9 Projection heads

Project image and pooled text into a contrastive space:  
$$
Z^t = g_t(T^{\text{pool}}) \in \mathbb{R}^{B \times d_o}  
$$
$$
Z^v = g_v(V) \in \mathbb{R}^{B \times d_o}  
$$
Recommended default:

- symmetric 2-layer MLP on both sides
    
- hidden dimension (D)
    
- output dimension (d_o = 256)
    
- GELU activation
    
- no dropout or small dropout (0.1)
    

Then L2 normalize:  
$$
\hat{Z}^t = \operatorname{norm}(Z^t)  
$$
$$
\hat{Z}^v = \operatorname{norm}(Z^v)  
$$
---

## 4.10 Contrastive alignment

Logit matrix:  
$$
\Lambda = \frac{\hat{Z}^t (\hat{Z}^v)^\top}{\tau_c} \in \mathbb{R}^{B \times B}  
$$
Text-to-image loss:  
$$
\mathcal{L}_{t2v} =
-\frac{1}{B}  
\sum_{i=1}^{B}  
\log  
\frac{\exp(\Lambda_{ii})}  
{\sum_{k=1}^{B}\exp(\Lambda_{ik})}  
$$
Image-to-text loss:  
$$
\mathcal{L}_{v2t} =
-\frac{1}{B}  
\sum_{i=1}^{B}  
\log  
\frac{\exp(\Lambda_{ii})}  
{\sum_{k=1}^{B}\exp(\Lambda_{ki})}  
$$
Main objective:  
$$
\mathcal{L}_{\text{InfoNCE}} =
\frac{1}{2}  
\left(  
\mathcal{L}_{t2v}  
+  
\mathcal{L}_{v2t}  
\right)  
$$
---

## 4.11 Gradient-flow summary

Gradients should flow through the full chain:  
$$
\mathcal{L}_{\text{InfoNCE}}
\rightarrow  
T^{\text{pool}}  
\rightarrow  
\beta  
\rightarrow  
Q  
\rightarrow  
\alpha  
\rightarrow  
\tilde{\Theta}_v  
\rightarrow  
\Theta_v  
$$
Specific notes:

- no stop-gradient in v1
    
- the prototype bank must receive nonzero gradient
    
- routing and pooling logits must remain differentiable
    
- no hard top-k or non-differentiable selection in v1
    

---

# 5. Design ledger: unresolved choices, options, defaults

## 5.1 Architecture-level decisions

|Design point|Why it matters|Reasonable options|Recommended default|Status|
|---|---|---|---|---|
|Global or sample-specific prototype bank|determines whether memory is reusable across dataset|global shared, sample-specific|global shared|fix now|
|Trainable vs frozen prototypes|affects whether bank adapts to task|trainable, frozen|trainable|fix now|
|Prototype contextualization form|controls prototype interaction and stability|overwrite, residual, off|residual|fix now|
|Single summary vs multi-head summary|controls capacity and complexity|single, multi-head|single|fix now|
|Token inclusion policy|strongly affects pooled text semantics|all tokens, content-only, EOS-only, CLS-only|content-only default|fix now|
|Symmetric vs asymmetric projectors|affects clarity and fairness of contrastive setup|symmetric, asymmetric|symmetric|fix now|
|Linear vs MLP projector|affects capacity and stability|linear, 2-layer MLP|2-layer MLP|fix now|
|Normalize projector outputs|essential for stable InfoNCE|yes, no|yes|fix now|

---

# 6. Backbone recommendations

## 6.1 Candidate visual encoders

1. CLIP ViT-B/32  
    Lightweight, stable, weaker granularity.
    
2. CLIP ViT-B/16  
    Best balance of strength and practicality.
    
3. CLIP RN50  
    Common baseline, but less aligned with transformer-style feature handling.
    
4. OpenCLIP or larger ViTs  
    Stronger ceiling, more compute, riskier for a first paper.
    

## 6.2 Candidate text encoders

1. paired CLIP text encoder  
    Cleanest default because it preserves the pretrained dual-encoder alignment.
    
2. BERT-base or RoBERTa-base  
    Strong token semantics, but introduces mismatch and extra confounds.
    
3. OpenCLIP text encoder  
    Appropriate if using a paired OpenCLIP visual backbone.
    

## 6.3 Recommendation table

|Regime|Visual encoder|Text encoder|Freeze policy|Purpose|
|---|---|---|---|---|
|Strong default|CLIP ViT-B/16|paired CLIP text encoder|frozen first, optional late partial unfreeze|main v1|
|Lightweight baseline|CLIP ViT-B/32|paired CLIP text encoder|frozen|fast debugging and early ablations|
|Stronger but riskier|larger OpenCLIP ViT|paired text encoder|partial finetuning|later final run|
|Classical baseline|CLIP RN50|paired CLIP text encoder|frozen|secondary comparison|

## 6.4 Pretraining and finetuning policy

Recommended for EMNLP-first experiments:

- use pretrained CLIP-style dual encoders
    
- freeze both backbones initially
    
- only unfreeze the last 1 to 2 blocks after the core module is verified
    

Rationale:

- isolates the contribution of the prototype-conditioned pooling
    
- reduces training instability
    
- keeps compute manageable
    
- avoids the criticism that gains come mainly from brute-force finetuning
    

---

# 7. Prototype bank design and initialization

## 7.1 Prototype interpretation

The prototype bank should be interpreted as:

- a shared latent visual memory
    
- a set of dataset-level visual anchors
    
- a routing basis that conditions text pooling
    

It should not be interpreted as a set of guaranteed semantic labels.

## 7.2 Number of prototypes

Recommended default:  
$$
N = 32  
$$
Primary ablation:  
$$
N \in {16, 32, 64}  
$$
Reasoning:

- too few prototypes underfit the diversity of visual space
    
- too many prototypes increase dead-prototype risk and introduce unnecessary complexity in v1
    

## 7.3 Prototype dimensionality

Recommended default:  
$$
D_p = D  
$$
This avoids introducing an extra projection inside the core method.

## 7.4 One bank or multiple banks

Recommended default:

- one shared bank only
    

Do not add:

- per-head banks
    
- hierarchical banks
    
- multi-bank memory variants
    

These can be future work but are unnecessary for the first technically sound paper.

---

## 7.5 Prototype initialization audit

### Option A. Random Gaussian init
$$
\theta_n \sim \mathcal{N}(0, \sigma^2 I)  
$$
Pros:

- trivial to implement
    

Cons:

- data-agnostic
    
- weak symmetry breaking
    
- can lead to redundant or dead prototypes
    

### Option B. Xavier init

Standard variance-preserving parameter initialization.

Pros:

- more stable than arbitrary Gaussian
    

Cons:

- still data-agnostic
    
- does not reflect image-space geometry
    

### Option C. Row-normalized random init

Sample randomly, then L2 normalize each prototype.

Pros:

- stable similarity scale from the start
    
- good fallback when no preprocessing is available
    

Cons:

- still not data-aware
    

### Option D. Initialize from sampled image embeddings

Procedure:

1. run a pretrained visual encoder on a subset of training images
    
2. sample (N) image embeddings
    
3. copy them into the bank
    
4. row-normalize
    

Pros:

- simple and data-aware
    
- gives immediately meaningful anchors
    

Cons:

- depends on random sample quality
    
- may overrepresent frequent patterns
    

### Option E. Initialize from k-means centroids over image embeddings

Procedure:

1. extract pretrained image embeddings on a subset of training data
    
2. run k-means with (k = N)
    
3. use cluster centroids as initialization
    
4. row-normalize
    

Pros:

- best symmetry breaking
    
- good coverage of pretrained image space
    
- fastest route to useful prototype usage
    

Cons:

- requires preprocessing
    

### Option F. Initialize from patch-token statistics

Possible but not recommended in v1 because the method currently routes from a global image embedding, not a set of visual tokens.

## 7.6 Recommended initialization

Default:

- **k-means centroids from pretrained global image embeddings, row-normalized**
    

Fallback:

- row-normalized random or Xavier init
    

Initialization ablation:

- normalized random
    
- sampled image embeddings
    
- k-means centroids
    

---

## 7.7 Dead-prototype detection

Define batch-average usage:  
$$
u_n = \frac{1}{B}\sum_{i=1}^B \alpha_{i,n}  
$$
Maintain a moving average:  
$$
\bar{u}_n^{(t)} = \rho \bar{u}_n^{(t-1)} + (1-\rho)u_n  
$$
A prototype is considered underused if:  
$$
\bar{u}_n < \epsilon  
$$
for a sustained number of steps.

Practical default:

- (\epsilon = 0.005)
    
- monitor over several hundred steps, not one batch
    

## 7.8 Reinitialization policy

Default:

- do **not** use automatic prototype reinitialization in the main training recipe
    

Emergency fallback:

- if many prototypes are persistently dead, reinitialize only the dead ones from sampled current image embeddings
    

This should not be part of the main method claim.

---

# 8. Similarity functions and normalization

A consistent normalization policy is one of the most important fixed decisions in the spec.

## 8.1 Prototype self-interaction

Candidate choices:

- raw dot product
    
- cosine similarity
    
- normalized dot product
    
- learned bilinear similarity
    

Recommended default:

- row-normalize prototypes before self-interaction
    
- use scaled dot product on normalized rows
    

Thus:  
$$
S_p = \hat{\Theta}_v \hat{\Theta}_v^\top  
$$
Why:

- avoids prototype norm domination
    
- stabilizes the contextualization step
    
- retains the parameter-free claim
    

## 8.2 Routing similarity

Candidate choices:

- raw dot product
    
- cosine similarity
    
- normalized dot product
    
- learned bilinear similarity
    

Recommended default:

- cosine similarity implemented as normalized dot product
    

Why:

- routing should measure directional compatibility, not norm magnitude
    
- this makes (\tau_p) interpretable
    

## 8.3 Text scoring similarity

Candidate choices:

- raw dot product
    
- cosine similarity
    
- additive attention scorer
    
- learned bilinear similarity
    

Recommended default:

- cosine similarity
    

Why:

- token scores become more interpretable
    
- reduces the risk that special tokens dominate purely via large norms
    

## 8.4 InfoNCE similarity

Recommended default:

- L2-normalized projector outputs with temperature-scaled cosine similarity
    

This should be treated as fixed for v1.

## 8.5 Summary normalization policy

Recommended v1:

- normalize prototypes for self-interaction
    
- normalize (V) and (\tilde{\Theta}_v) for routing similarity
    
- normalize (Q) and (H) for token scoring similarity
    
- normalize (Z^t) and (Z^v) before InfoNCE
    
- keep the feature tensors themselves unconstrained as parameters, but normalize them at similarity computation time
    

---

# 9. Temperatures and scaling

Use separate temperatures for each module:

- (\tau_p): routing temperature
    
- (\tau_t): token pooling temperature
    
- (\tau_c): contrastive temperature
    

## 9.1 Recommended defaults

For v1:

- (\tau_p = 0.07)
    
- (\tau_t = 0.07)
    
- (\tau_c): learnable, initialized to the equivalent of (0.07)
    

## 9.2 Learnable vs fixed

Recommended:

- (\tau_p): fixed in v1
    
- (\tau_t): fixed in v1
    
- (\tau_c): learnable
    

Why:

- routing and token-pooling temperatures are easier to debug when fixed
    
- contrastive temperature often benefits from learning
    

## 9.3 Parameterization

If temperature is learned, optimize it via log-parameterization:  
$$
\tau = \exp(\phi)  
$$
or equivalently optimize a logit scale parameter and exponentiate it.

This is preferred over directly optimizing (\tau).

## 9.4 Self-attention scaling

Keep the standard:  
$$
\frac{1}{\sqrt D}  
$$
inside prototype self-attention.

Do not add a separate self-attention temperature in v1 unless the prototype contextualization step becomes demonstrably too sharp or too flat.

---

# 10. Loss design

## 10.1 Main contrastive loss

The main objective is the symmetric InfoNCE loss defined in Section 4.10:  
$$
\mathcal{L}_{\text{InfoNCE}}  
$$
This is the only indispensable training objective.

---

## 10.2 Prototype diversity regularization

Recommended form:

First normalize prototypes row-wise:  
$$
\hat{\Theta}_v = \operatorname{norm}(\Theta_v)  
$$
Then penalize off-diagonal similarity:  
$$
\mathcal{L}_{\text{div}} =
\left|  
\hat{\Theta}_v \hat{\Theta}_v^\top - I  
\right|_F^2  
$$
Purpose:

- discourages prototype redundancy
    
- reduces collapse into nearly identical anchors
    

Side effect:

- if over-weighted, it may over-separate prototypes and hurt performance
    

Recommended coefficient:  
$$
\lambda_{\text{div}} = 10^{-2}  
$$
Search range:  
$$
10^{-4} \text{ to } 10^{-1}  
$$
Status:

- include in the main v1 recipe
    

---

## 10.3 MoE-inspired balancing loss

Let:  
$$
\bar{\alpha}_n = \frac{1}{B}\sum_{i=1}^B \alpha_{i,n}  
$$
Simple balancing loss:  
$$
\mathcal{L}_{\text{bal}} =
\sum_{n=1}^{N}  
\left(  
\bar{\alpha}_n - \frac{1}{N}  
\right)^2  
$$
Purpose:

- discourages global routing collapse onto a few prototypes
    

Side effect:

- can force artificial use of unhelpful prototypes
    
- may conflict with natural data skew
    

Recommended coefficient:  
$$
\lambda_{\text{bal}} \in [10^{-4}, 10^{-2}]  
$$
Status:

- not part of minimal v1
    
- add only if prototype underuse is observed
    

---

## 10.4 Optional entropy control on routing

Per-sample routing entropy:  
$$
H(\alpha_i) =
-\sum_{n=1}^{N}  
\alpha_{i,n}\log \alpha_{i,n}  
$$
This can be used to encourage softer or sharper routing, but it overlaps strongly with the role of (\tau_p).

Status:

- ablation only
    
- not recommended in v1
    

---

## 10.5 Optional entropy control on token pooling

Per-sample token entropy:  
$$
H(\beta_i) =
-\sum_{j=1}^{L}  
\beta_{i,j}\log \beta_{i,j}  
$$
This can be used only if token pooling becomes degenerate.

Status:

- ablation only
    
- not recommended in v1
    

---

## 10.6 Final training objective

### Minimal clean v1
$$
\mathcal{L} =
\mathcal{L}_{\text{InfoNCE}}  
+  
\lambda_{\text{div}} \mathcal{L}_{\text{div}}  
$$
### Stronger v2
$$
\mathcal{L} =
\mathcal{L}_{\text{InfoNCE}}  
+  
\lambda_{\text{div}} \mathcal{L}_{\text{div}}  
+  
\lambda_{\text{bal}} \mathcal{L}_{\text{bal}}  
$$
---

# 11. Full hyperparameter inventory

|Hyperparameter|Role|Recommended default|Search range|Sensitivity|Status|
|---|---|--:|---|---|---|
|visual backbone|image encoder|CLIP ViT-B/16|ViT-B/32, RN50|high|fix now|
|text backbone|token encoder|paired CLIP text|paired OpenCLIP text|high|fix now|
|pretraining regime|initialization of encoders|pretrained|none|very high|fix now|
|visual freeze policy|stability vs capacity|frozen first|last 1 block, last 2 blocks|very high|fix now|
|text freeze policy|stability vs capacity|frozen first|last 1 block, full finetune|high|fix now|
|(D)|hidden feature dimension|backbone-native|backbone-dependent|medium|fix now|
|(N)|number of prototypes|32|16, 32, 64|high|fix now|
|prototype dimension|bank feature dimension|(D)|(D/2), (D)|medium|fix now|
|prototype contextualization|prototype interaction|on with residual|off, overwrite|high|fix now|
|prototype init type|initial geometry of bank|k-means centroids|normalized random, sampled embeddings|very high|fix now|
|prototype init scale|used if random init|0.02 then row norm|0.01 to 0.1|medium|fix now|
|normalize prototypes at init|stable similarity|yes|yes/no|high|fix now|
|self-similarity type|prototype interaction score|normalized dot|raw dot|high|fix now|
|routing similarity|image-to-prototype score|cosine|raw dot|very high|fix now|
|text scoring similarity|query-to-token score|cosine|raw dot, additive scorer|very high|fix now|
|InfoNCE similarity|final alignment|cosine on normalized outputs|raw dot|very high|fix now|
|(\tau_p)|routing softness|0.07|0.03 to 0.2|high|fix now|
|(\tau_t)|token-pooling softness|0.07|0.03 to 0.2|very high|fix now|
|(\tau_c)|contrastive scale|learnable, init from 0.07|0.01 to 0.1 if fixed|very high|fix now|
|special-token policy|token inclusion|exclude by default|include all, content-only, EOS-only|very high|fix now|
|max text length|text truncation budget|backbone default|64, 77, 128|medium|fix now|
|truncation strategy|preserve useful content|tokenizer default, preserve prefix|head-only, head+tail|medium|fix now|
|image resolution|backbone input|backbone default|224, 336|medium|fix now|
|projector type|projection head|symmetric 2-layer MLP|linear, 3-layer MLP|high|fix now|
|projector hidden size|projector capacity|(D)|(D/2), (D), (2D)|medium|fix now|
|projector output size|contrastive dimension|256|128, 256, 512|medium|fix now|
|projector activation|nonlinearity|GELU|ReLU, SiLU|low|fix now|
|projector dropout|regularization|0.0 to 0.1|0.0 to 0.3|low|can ablate|
|normalize projector outputs|stable InfoNCE|yes|yes/no|very high|fix now|
|optimizer|optimization|AdamW|AdamW, Lion|high|fix now|
|LR backbone|backbone update speed|(1e{-5}) if partially unfrozen|(1e{-6}) to (5e{-5})|very high|fix now|
|LR prototypes|bank learning speed|(1e{-3})|(1e{-4}) to (5e{-3})|high|fix now|
|LR projectors|projector learning speed|(1e{-3})|(1e{-4}) to (5e{-3})|high|fix now|
|LR temperature|logit-scale learning speed|(1e{-4}) to (1e{-3})|(1e{-5}) to (1e{-3})|medium|fix now|
|weight decay backbone|regularization|0.01|0.0 to 0.05|medium|fix now|
|weight decay prototypes|bank regularization|0.01|0.0 to 0.1|medium|fix now|
|weight decay projectors|head regularization|0.05|0.01 to 0.1|medium|fix now|
|AdamW beta1|optimizer coefficient|0.9|fixed|low|fix now|
|AdamW beta2|optimizer coefficient|0.98 or 0.999|fixed|low|fix now|
|grad clipping|prevent instability|1.0|0.5 to 5.0|medium|fix now|
|scheduler|LR decay|cosine|cosine, linear|medium|fix now|
|warmup ratio|early training stability|5%|2% to 10%|medium|fix now|
|training epochs|total training|10 to 30|5 to 50|high|fix now|
|batch size|per-device|memory-limited|16 to 128|high|fix now|
|effective batch size|contrastive negatives|128 or 256|64 to 1024|very high|fix now|
|mixed precision|speed and memory|on|fp16, bf16|medium|fix now|
|grad accumulation|larger effective batch|as needed|1 to 8|medium|fix now|
|EMA|optional stabilization|off|on/off|low|can ablate|
|random seed|reproducibility|3 seeds|1, 3, 5|high|fix now|
|(\lambda_{\text{div}})|diversity regularization|(1e{-2})|(1e{-4}) to (1e{-1})|high|fix now|
|(\lambda_{\text{bal}})|balancing regularization|0 in v1|(1e{-4}) to (1e{-2})|high|can ablate|
|routing entropy coefficient|control alpha sharpness|0|(1e{-5}) to (1e{-2})|medium|can ablate|
|token entropy coefficient|control beta sharpness|0|(1e{-5}) to (1e{-2})|medium|can ablate|
|tokenizer|text preprocessing|backbone-native|backbone-dependent|high|fix now|
|caption cleaning|text normalization|minimal|lowercase, punctuation cleanup|medium|fix now|
|image augmentation|visual robustness|light, CLIP-consistent|resize-crop, flip|medium|fix now|
|multiple captions per image|data usage|dataset default first|one-caption sampling, all-captions use|medium|fix now|
|train-val split|benchmark protocol|standard split|custom if justified|very high|fix now|
|evaluation metrics|retrieval|R@1, R@5, R@10, MedR, MeanR|benchmark standard|very high|fix now|
|report variance|reliability|mean ± std over 3 seeds|1, 3, 5 runs|high|fix now|

---

# 12. Training protocol

## 12.1 Default recipe for v1

### Model

- visual encoder: pretrained CLIP ViT-B/16
    
- text encoder: paired pretrained CLIP text encoder
    
- freeze both encoders initially
    
- prototype bank: (N=32), dimension (D)
    
- prototype init: k-means centroids from pretrained image embeddings
    
- contextualization: normalized parameter-free self-attention with residual
    
- routing similarity: cosine
    
- token scoring similarity: cosine
    
- projector: symmetric 2-layer MLP
    
- output dimension: (256)
    

### Loss
$$
\mathcal{L} =
\mathcal{L}_{\text{InfoNCE}}  
+  
10^{-2}\mathcal{L}_{\text{div}}  
$$
### Optimizer

- AdamW
    
- separate parameter groups for backbone, prototypes, projectors, temperatures
    

Recommended initial learning rates:

- backbone: 0 if frozen
    
- prototypes: (1e{-3})
    
- projectors: (1e{-3})
    
- learnable contrastive temperature: (1e{-4}) or (1e{-3})
    

### Schedule

- cosine scheduler
    
- 5% warmup
    
- mixed precision on
    
- global grad norm clipping at 1.0
    

---

## 12.2 Parameter-group recommendation

Use distinct optimizer groups:

1. frozen or partially unfrozen backbone parameters
    
2. prototype bank parameters
    
3. projector parameters
    
4. learnable temperature parameters
    

This is important because the prototype bank often needs a higher learning rate than the backbone.

---

# 13. Evaluation protocol

## 13.1 Primary metrics

Report both directions:

- text-to-image retrieval
    
- image-to-text retrieval
    

Metrics:

- Recall@1
    
- Recall@5
    
- Recall@10
    
- Median Rank
    
- Mean Rank
    

## 13.2 Reporting protocol

- report mean and standard deviation over 3 seeds for the main configuration
    
- keep evaluation settings identical across ablations
    
- use the benchmark-standard train/val/test split
    
- clearly separate:
    
    - zero-shot baseline
        
    - finetuned baseline
        
    - proposed model
        

## 13.3 Required baselines

At minimum implement:

1. EOS-only pooled text baseline
    
2. content-token mean-pooling baseline
    
3. no-prototype-bank variant
    
4. no-contextualization variant
    

These are necessary to make the mechanism convincing.

---

# 14. Failure modes, monitoring, and debugging

## 14.1 Main failure modes

### Prototype collapse

Symptom:

- many prototypes have pairwise cosine similarity close to 1
    

Likely cause:

- weak initialization
    
- no diversity pressure
    
- over-strong shared gradients
    

Fix:

- improve initialization
    
- add or strengthen diversity loss
    
- verify normalized similarities
    

### Dead prototypes

Symptom:

- some prototypes receive near-zero routing mass over long periods
    

Likely cause:

- low routing temperature
    
- poor initialization
    
- severe usage imbalance
    

Fix:

- increase (\tau_p)
    
- add balancing loss
    
- switch to k-means init
    

### Uniform routing

Symptom:

- (\alpha_i) is almost uniform for most samples
    

Likely cause:

- high (\tau_p)
    
- weak prototype separation
    

Fix:

- lower (\tau_p)
    
- verify bank diversity
    
- reduce over-regularization
    

### Overly sharp routing

Symptom:

- (\alpha_i) is nearly one-hot very early
    

Likely cause:

- low (\tau_p)
    
- uncontrolled norms
    
- poor temperature scaling
    

Fix:

- raise (\tau_p)
    
- verify normalization
    
- clip gradients
    

### Token pooling dominated by special tokens

Symptom:

- EOS or SOS consistently gets most of the weight
    

Likely cause:

- special tokens have high norms or shortcut semantics
    

Fix:

- exclude them in the default model
    
- verify token masking
    
- use cosine token scoring
    

### Projector collapse

Symptom:

- projected features have very low variance
    
- similarity matrix becomes nearly constant
    

Fix:

- lower projector LR
    
- use 2-layer MLP instead of a weak linear head
    
- verify contrastive temperature behavior
    

### Temperature instability

Symptom:

- learned (\tau_c) explodes or collapses
    

Fix:

- log-parameterize
    
- clamp if needed
    
- give temperature its own LR
    

---

## 14.2 Debugging checklist

### Shape checks

Assert:

- (V): ([B, D])
    
- (H): ([B, L, D])
    
- (\Theta_v): ([N, D])
    
- prototype similarity matrix: ([N, N])
    
- routing logits: ([B, N])
    
- (\alpha): ([B, N])
    
- (Q): ([B, D])
    
- token scores: ([B, L])
    
- (\beta): ([B, L])
    
- pooled text: ([B, D])
    

### Softmax checks

Verify:

- (\sum_n \alpha_{i,n} = 1)
    
- (\sum_j \beta_{i,j} = 1) over valid tokens only
    

### Mask checks

Verify:

- padded tokens get exactly zero beta after softmax
    
- excluded special tokens get exactly zero beta after softmax
    

### Normalization checks

Track:

- norms of (V), (Q), (H), (\Theta_v), (Z^t), (Z^v)
    

### Gradient checks

Track gradient norms for:

- prototype bank
    
- projectors
    
- unfrozen backbone blocks
    
- temperature parameters
    

### Distribution checks

Track:

- mean entropy of (\alpha)
    
- mean entropy of (\beta)
    
- prototype usage histogram
    
- top token identity histogram
    
- q-norm distribution
    
- pooled-text norm distribution
    

### Geometry checks

Visualize:

- pairwise cosine matrix of prototypes
    
- routing usage across prototypes
    
- cosine similarity matrix of projected features
    
- InfoNCE positive vs negative logit distributions
    

### Sanity experiments

Run before full training:

1. overfit a tiny subset of 32 to 128 samples
    
2. compare bank-on vs bank-off
    
3. inspect top beta-weighted tokens manually
    
4. verify that prototype usage is not trivially uniform or collapsed
    
5. confirm that no masked token receives nonzero attention
    

---

# 15. Ablation plan

## 15.1 Essential ablations

These are the most important for a convincing paper.

### A. Token pooling policy

- content-only pooling
    
- content + special tokens
    
- EOS-only
    
- mean-pooling
    
- CLS-only if applicable
    

### B. Prototype-bank utility

- no bank
    
- bank without contextualization
    
- bank with contextualization
    

### C. Number of prototypes

- (N = 16, 32, 64)
    

### D. Diversity regularization

- off
    
- on
    

### E. Similarity choice

- cosine vs raw dot for routing
    
- cosine vs raw dot for token scoring
    

These ablations directly support the core mechanism claim.

---

## 15.2 Nice-to-have ablations

- prototype initialization variants
    
- balancing loss on/off
    
- projector depth
    
- normalization variants
    
- frozen vs partial-unfreeze backbone
    

---

## 15.3 Risky or expensive ablations

- larger backbones
    
- multi-head prototype summaries
    
- learned bilinear scoring
    
- more than one prototype bank
    

These should be postponed unless the paper is already strong and stable.

---

# 16. Algorithmic summary

## Algorithm 1: Forward pass

Given image batch (X), token batch (T), mask (M), prototype bank (\Theta_v):

1. compute global image embeddings:  
$$
    V = f_v(X)  
$$
2. compute text token features:  
$$
    H = f_t(T, M)  
$$
3. contextualize prototype bank:  
$$
    \hat{\Theta}_v = \operatorname{norm}(\Theta_v)  
$$
$$
    P_p = \operatorname{softmax}\left(\frac{\hat{\Theta}_v \hat{\Theta}_v^\top}{\sqrt D}, \text{dim}=-1\right)  
$$
$$
    \tilde{\Theta}_v = \Theta_v + P_p \Theta_v  
$$
4. compute routing weights:  
$$
    \alpha = \operatorname{softmax}\left(\frac{\operatorname{sim}(V, \tilde{\Theta}_v)}{\tau_p}, \text{dim}=-1\right)  
$$
5. compute image-conditioned summary:  
$$
    Q = \alpha \tilde{\Theta}_v  
$$
6. compute token scores:  
$$
    S_t = \frac{\operatorname{sim}(Q, H)}{\tau_t}  
$$
7. apply masking and obtain token weights:  
$$
    \beta = \operatorname{masked_softmax}(S_t, \text{dim}=-1)  
$$
8. pool text:  
$$
    T^{\text{pool}} = \sum_j \beta_j h_j  
$$
9. project and normalize:  
$$
    \hat{Z}^t = \operatorname{norm}(g_t(T^{\text{pool}}))  
$$
$$
    \hat{Z}^v = \operatorname{norm}(g_v(V))  
$$
10. compute symmetric InfoNCE
    
11. add prototype diversity regularization
    
12. backpropagate and update trainable parameters
    

---

# 17. Recommended v1 implementation recipe

The smallest technically sound version to build first is:

## Model

- pretrained CLIP ViT-B/16 visual encoder
    
- pretrained paired CLIP text encoder
    
- both frozen initially
    
- one shared prototype bank with (N=32)
    
- k-means centroid initialization from pretrained image embeddings
    
- residual prototype contextualization
    
- cosine routing
    
- cosine token scoring
    
- content-token-only pooling by default
    
- symmetric 2-layer MLP projectors
    
- normalized InfoNCE outputs
    

## Loss
$$
\mathcal{L} =
\mathcal{L}_{\text{InfoNCE}}  
+  
10^{-2}\mathcal{L}_{\text{div}}  
$$
## Training

- AdamW
    
- prototypes and projectors trained from scratch
    
- contrastive temperature learnable
    
- mixed precision
    
- cosine schedule
    
- 3 random seeds
    

## Immediate next extensions

1. add balancing loss if prototype underuse appears
    
2. partially unfreeze the last visual block
    
3. run initialization ablations
    

## Postpone

- multi-head summaries
    
- multiple memory banks
    
- sparse routing
    
- learned bilinear scorers
    
- patch-token conditioning
    

---

# 18. Open questions requiring mentor confirmation

These are the few remaining choices that are still genuinely open at the design level.

1. Should prototype contextualization be presented as **residual** or stay in the original overwrite form?
    
2. Should the paper preserve a strict **parameter-free contextualization** claim?
    
3. Is the default token-pooling policy intended to **exclude** special tokens, with inclusion only as ablation?
    
4. Should the first main result use **fully frozen backbones** for cleaner attribution, or partial finetuning for stronger numbers?
    
5. Is the prototype bank framed as a **latent memory** only, or does the paper want stronger interpretability claims?
    
6. Should balancing loss remain **optional** unless dead prototypes appear, or be part of the main recipe from the start?
    
7. Is the intended comparison set mainly:
    
    - EOS or CLS pooling baselines, and lightweight interaction baselines,  
        or also
        
    - heavier cross-attention baselines in the first paper?
        
8. Should (N=32) be treated as the fixed default, with (16) and (64) as essential ablations?
    
9. Is a single global image vector sufficient for the EMNLP-first version, or is there mentor interest in migrating later to patch-level visual evidence?
    

