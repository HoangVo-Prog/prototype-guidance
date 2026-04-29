## Integrated ITSELF Host with Prototype-Mediated Semantic Structure, Objective Design, Training Strategy, and Inference Pipeline

  

### 1. Problem setting

  

Each training sample is a triplet

  

$$  

(x_i, c_i, y_i)  

$$

  

where:

  

- $x_i$ is the image

- $c_i$ is the caption

- $y_i$ is the person identity label

- $i$ indexes a sample in the mini-batch

  

The overall goal is to build a unified text-based person search model in which:

  

1. ITSELF remains the main retrieval host

2. the host keeps its original global and local alignment path

3. the prototype branch no longer acts as a second retrieval scorer

4. the prototype branch instead acts as a semantic structure module that regularizes the host-side representation space and the surrogate text construction process

5. the surrogate text object remains important and is still tied to an exact diagonal teacher through a diagonal fidelity loss

  

The central idea of the revised method is that ITSELF should continue to carry the main retrieval burden, while the prototype subsystem should provide semantic organization rather than an additive retrieval score. In other words, the prototype branch is reinterpreted from a retrieval enhancement branch into a semantic anchor system for the host.

  

---

  

### 2. Model overview

  

The revised integrated model contains the following conceptual components:

  

- an ITSELF host image encoder

- an ITSELF host text encoder

- the ITSELF GRAB local alignment path

- a prototype bank induced from the current representation space by K-Means

- an optional lightweight contextualization step on top of the induced prototypes

- an image-conditioned routing mechanism

- a prototype-conditioned text basis construction module

- a surrogate image-conditioned text construction module

- an exact diagonal text teacher object for fidelity

- a semantic structure regularizer based on prototype assignments and Prototype Back Translation

  

The key modeling principle is:

  

1. use ITSELF to produce strong host visual and textual representations

2. preserve the original ITSELF host score path and let it remain the deployed retrieval scorer

3. use the host global image feature to route over semantic prototypes

4. use the resulting image-conditioned summary to construct a surrogate text object

5. use the prototype system to regularize representation grouping instead of producing an additive retrieval score

6. use diagonal fidelity to keep the surrogate text semantically grounded

  

So the prototype subsystem is no longer defined as an auxiliary retrieval expert. It is defined as a semantic structure mechanism for organizing the host-side geometry and stabilizing image-conditioned text construction.

  

---

  

### 3. ITSELF host branch

  

The host framework is ITSELF. It uses CLIP ViT-B/16 encoders for both modalities and adds GRAB, with MARS and ATS, to learn fine-grained local correspondences. The host is trained with a dual-loss design consisting of a global loss and a local loss, and the host inference score combines global and local similarity.

  

Given an image $x_i$, the ITSELF image encoder produces

  

$$  

V_i = f_v^{\text{host}}(x_i) = \{v_i^{\text{global}}, v_i^{\text{local}}\}  

$$

  

where:

  

- $v_i^{\text{global}} \in \mathbb{R}^{D}$ is the global image feature

- $v_i^{\text{local}} = \{v_{i,r}^{\text{local}}\}_{r=1}^{R}$ are local patch features

  

Given a caption $c_j$, the ITSELF text encoder produces

  

$$  

T_j = f_t^{\text{host}}(c_j) = \{t_j^{\text{global}}, t_j^{\text{local}}\}  

$$

  

where:

  

- $t_j^{\text{global}} \in \mathbb{R}^{D}$ is the global text feature

- $t_j^{\text{local}} = \{t_{j,\ell}^{\text{local}}\}_{\ell=1}^{L_j}$ are token-level text features

  

For the prototype branch, we denote the token-level text states by

  

$$  

H_j = [h_{j,1}, h_{j,2}, \dots, h_{j,L_j}] \in \mathbb{R}^{L_j \times D}  

$$

  

with

  

$$  

h_{j,\ell} = t_{j,\ell}^{\text{local}}  

$$

  

The ITSELF host branch also constructs guided local features through GRAB. These host components are preserved as part of the base system and are not replaced by the prototype subsystem.

  

---

  

### 4. Host retrieval path

  

Let the host global projection heads produce

  

$$  

z_i^{v,\text{host}} = g_v^{\text{host}}(v_i^{\text{global}})  

\qquad\text{and}\qquad  

z_j^{t,\text{host}} = g_t^{\text{host}}(t_j^{\text{global}})  

$$

  

Let the GRAB local branch produce local guided representations for image and text, denoted abstractly by

  

$$  

\bar{v}_i^{\text{loc,host}}  

\qquad\text{and}\qquad  

\bar{t}_j^{\text{loc,host}}  

$$

  

The host defines a global similarity

$$  

s_{ij}^{\text{global,host}}

=

\operatorname{sim}\left(z_i^{v,\text{host}}, z_j^{t,\text{host}}\right)  

$$

  

and a local similarity

  

$$  

s_{ij}^{\text{local,host}}

=

\operatorname{sim}\left(\bar{v}_i^{\text{loc,host}}, \bar{t}_j^{\text{loc,host}}\right)  

$$

  

The ITSELF host score is then

  

$$  

s_{ij}^{\text{host}}

=

\lambda_s s_{ij}^{\text{global,host}}  

+  

(1-\lambda_s)s_{ij}^{\text{local,host}}  

$$

  

This host score remains the deployed retrieval score of the revised integrated system.

  

---

  

### 5. From prototype retrieval branch to prototype semantic structure branch

  

In the earlier formulation, the prototype branch was attached on top of the host in order to produce an additional pairwise prototype score and then fuse this score with the host score.

  

In the revised formulation, the prototype branch is redefined in the following way:

  

1. it still uses the host global image feature as the routing signal

2. it still uses the host token-level text states as the text construction substrate

3. it still constructs an image-conditioned surrogate text object

4. it no longer defines an additive prototype retrieval score that competes with the host score

5. it instead provides semantic anchors and semantic grouping targets that regularize the host-side representation geometry

6. it preserves the surrogate-exact diagonal relation through the diagonal fidelity loss

  

So the host remains the only retrieval scorer, while the prototype subsystem becomes a semantic organizer.

  

---

  

### 6. Prototype bank as semantic anchors

  

The prototype bank is no longer treated as a learnable parameter bank updated directly by SGD. Instead, it is induced from the current representation space by K-Means and updated by **periodic recomputation**.

  

Let the prototype-input image feature be

  

$$  

u_i^{I} = g_v^{\text{proto}}(v_i^{\text{global}}) \in \mathbb{R}^{d_h}  

$$

  

and let the prototype-input text feature for the matched pair be:

  

$$  

u_i^{T,\text{exact}} = g_t^{\text{proto}}\left(t_{i \mid i}^{\text{exact}}\right) \in \mathbb{R}^{d_h}  

$$

  

where $g_v^{\text{proto}}$ and $g_t^{\text{proto}}$ denote prototype-side projection heads.

  

At prototype update step $r$, we construct image and text prototypes by K-Means on detached features:

  

$$  

P_r^{I} =  

\begin{bmatrix}  

p_{r,1}^{I} \  

p_{r,2}^{I} \  

\vdots \  

p_{r,N}^{I}  

\end{bmatrix}  

\in \mathbb{R}^{N \times d_h}  

\qquad\text{and}\qquad  

P_r^{T} =  

\begin{bmatrix}  

p_{r,1}^{T} \  

p_{r,2}^{T} \  

\vdots \  

p_{r,N}^{T}  

\end{bmatrix}  

\in \mathbb{R}^{N \times d_h}  

$$

  

with

  

$$  

P_r^{I} = \operatorname{KMeans}(\operatorname{sg}({\nu_i^{I}}))  

\qquad\text{and}\qquad  

P_r^{T} = \operatorname{KMeans}(\operatorname{sg}({\nu_i^{T,\text{exact}}}))  

$$

  

where $\operatorname{sg}(\cdot)$ denotes stop-gradient or detach.

  

The meaning of this design is that semantic structure is induced from the data distribution in the current feature space rather than learned directly by gradient updates on prototype parameters.

  

---

  

### 7. Prototype contextualization as lightweight post-processing

  

If prototype contextualization is used, it is applied as a lightweight post-processing step on top of the recomputed semantic anchors.

  

For example, the contextualized prototypes can be formed by

  

$$  

\tilde{P}_r^{I}

=

\operatorname{softmax}  

\left(  

\frac{P_r^{I}(P_r^{I})^\top}{\sqrt{d_h}}  

\right) P_r^{I}  

$$

  

and

  

$$  

\tilde{P}_r^{T}

=

\operatorname{softmax}  

\left(  

\frac{P_r^{T}(P_r^{T})^\top}{\sqrt{d_h}}  

\right) P_r^{T}  

$$

  

To preserve the semantic purity of the K-Means anchors, a residual version is preferred:

  

$$  

\bar{P}_r^{I} = P_r^{I} + \alpha_c \tilde{P}_r^{I}  

\qquad\text{and}\qquad  

\bar{P}_r^{T} = P_r^{T} + \alpha_c \tilde{P}_r^{T}  

$$

  

where $\alpha_c$ is a small constant or a fixed scalar.

  

If contextualization is disabled, we simply use

  

$$  

\bar{P}_r^{I} = P_r^{I}  

\qquad\text{and}\qquad  

\bar{P}_r^{T} = P_r^{T}  

$$

  

The important modeling principle is that semantic structure is still attributed to the recomputed K-Means anchors, while contextualization is only a lightweight refinement layer.

  

---

  

### 8. Image-conditioned routing

  

The routing signal is taken from the host global image feature through the image-side semantic anchors.

  

For image $x_i$, the routing weights are

  

$$  

\alpha_{i,n}

=

\frac{  

\exp\left(\operatorname{sim}(\nu_i^{I}, \bar{p}_{r,n}^{I})/\tau_p\right)  

}{  

\sum_{m=1}^{N}  

\exp\left(\operatorname{sim}(\nu_i^{I}, \bar{p}_{r,m}^{I})/\tau_p\right)  

}  

$$

  

where $\tau_p$ is the routing temperature and $\bar{p}_{r,n}^{I}$ is the $n$-th image-side contextualized prototype.

  

The routing weights satisfy

  

$$  

\sum_{n=1}^{N}\alpha_{i,n}=1  

$$

  

Using these routing weights, the prototype subsystem forms an image-conditioned summary

  

$$  

q_i

=

\sum_{n=1}^{N}  

\alpha_{i,n}\bar{p}_{r,n}^{I}  

\in \mathbb{R}^{d_h}  

$$

  

This $q_i$ is the image-conditioned semantic summary associated with image $i$.

  

---

  

### 9. Prototype-conditioned text basis construction

  

For each caption $c_j$, the host text encoder provides token states

  

$$  

H_j = [h_{j,1}, h_{j,2}, \dots, h_{j,L_j}]  

$$

  

For each text-side prototype $\bar{p}_{r,n}^{T}$, the branch computes token scores

  

$$  

u_{j,n,\ell}

=

\frac{  

(\bar{p}_{r,n}^{T})^\top h_{j,\ell}  

}{\tau_b}  

$$

  

The corresponding token weights are

  

$$  

\gamma_{j,n,\ell}

=

\frac{  

\exp\left(\nu_{j,n,\ell}\right)  

}{  

\sum_{r=1}^{L_j}\exp\left(\nu_{j,n,r}\right)  

}  

$$

  

The prototype-conditioned basis vector for caption $j$ under prototype $n$ is

  

$$  

b_{j,n}

=

\sum_{\ell=1}^{L_j}  

\gamma_{j,n,\ell} h_{j,\ell}  

\in \mathbb{R}^{D}  

$$

  

Collecting all prototype-conditioned basis vectors gives the caption basis bank

  

$$  

B_j =  

\begin{bmatrix}  

b_{j,1} \  

b_{j,2} \  

\vdots \  

b_{j,N}  

\end{bmatrix}  

\in \mathbb{R}^{N \times D}  

$$

  

So each caption is represented not by a single fixed prototype-agnostic vector in the prototype subsystem, but by a bank of $N$ text bases induced by the current semantic anchor set.

  

---

  

### 10. Surrogate prototype text object

  

For image $i$ and caption $j$, the prototype subsystem constructs a surrogate image-conditioned text object by mixing the caption basis bank with the routing weights of image $i$:

  

$$  

\hat{t}_{j \mid i}^{\text{proto}}

=

\sum_{n=1}^{N}  

\alpha_{i,n} b_{j,n}  

$$

  

The prototype text projector then produces

  

$$  

\hat{z}_{j \mid i}^{t,\text{proto}}

=

g_t^{\text{proto}}\left(\hat{t}_{j \mid i}^{\text{proto}}\right)  

$$

  

This surrogate text object remains the central text-side object produced by the prototype subsystem.

  

---

  

### 11. Exact diagonal teacher object for fidelity

  

Although the prototype subsystem no longer defines a separate retrieval score, it still uses an exact diagonal teacher on matched pairs to keep the surrogate semantically grounded.

  

For the matched pair $(i,i)$, token scores under image-conditioned summary $q_i$ are

  

$$  

s_{i,\ell}^{\text{exact}}

=

\frac{  

q_i^\top h_{i,\ell}  

}{\tau_t}  

$$

  

The corresponding token weights are

  

$$  

\beta_{i,\ell}^{\text{exact}}

=

\frac{  

\exp\left(s_{i,\ell}^{\text{exact}}\right)  

}{  

\sum_{r=1}^{L_i}\exp\left(s_{i,r}^{\text{exact}}\right)  

}  

$$

  

The exact diagonal text object is

  

$$  

t_{i \mid i}^{\text{exact}}

=

\sum_{\ell=1}^{L_i}  

\beta_{i,\ell}^{\text{exact}} h_{i,\ell}  

$$

  

and its projected teacher embedding is

  

$$  

z_{i \mid i}^{t,\text{exact}}

=

g_t^{\text{proto}}\left(t_{i \mid i}^{\text{exact}}\right)  

$$

  

This exact diagonal object is used as the semantic teacher for the surrogate branch and also provides the text-side semantic feature used in prototype recomputation.

  

---

  

### 12. Diagonal fidelity loss

  

The surrogate branch is tied to the exact diagonal teacher through a diagonal fidelity loss.

  

The diagonal surrogate embedding is

  

$$  

\hat{z}_{i \mid i}^{t,\text{proto}}

=

g_t^{\text{proto}}\left(\hat{t}_{i \mid i}^{\text{proto}}\right)  

$$

  

The exact diagonal teacher embedding is

  

$$  

z_{i \mid i}^{t,\text{exact}}

=

g_t^{\text{proto}}\left(t_{i \mid i}^{\text{exact}}\right)  

$$

  

The fidelity loss is

  

$$

M_{ik} = \cos\left(\hat{z}_{i|i}^{t,\text{proto}}, \ \mathrm{sg}\left(z_{k|k}^{t,\text{exact}}\right)\right)

$$

$$

\mathcal{L}_{\text{diag-rel}}^{\text{sym}} = \frac{1}{2} \left( \mathcal{L}_{\text{row}} + \mathcal{L}_{\text{col}} \right)

$$

$$

\mathcal{L}_{\text{row}} = - \frac{1}{B} \sum_i \log \frac{\exp\left(M_{ii} / \tau_d\right)}{\sum_k \exp\left(M_{ik} / \tau_d\right)}

$$

$$

\mathcal{L}_{\text{col}} = - \frac{1}{B} \sum_i \log \frac{\exp\left(M_{ii} / \tau_d\right)}{\sum_k \exp\left(M_{ki} / \tau_d\right)}

$$

  

In the revised method, this term remains fixed because it is the key loss that keeps the surrogate construction semantically anchored to the exact image-conditioned diagonal teacher.

  

---

  

### 13. Semantic representation grouping view

  

The revised method separates two goals:

  

1. representation alignment for retrieval, which remains the responsibility of the host objective

2. representation grouping for semantic organization, which becomes the responsibility of the prototype subsystem

  

The semantic structure loss therefore should not be defined as an additional retrieval loss. Instead, it should use prototype assignments and translated centroids to regularize the geometry of:

  

- image-side semantic features

- surrogate text semantic features

  

This follows the view that prototypes are semantic anchors rather than a second score path.

  

---

  

### 14. Prototype assignments

  

Using the base recomputed prototypes before contextualization, each image and matched exact text feature receives a prototype assignment.

  

For the image side, let the assigned prototype index be

  

$$  

a_i^{I} = \arg\max_{k} \operatorname{sim}(\nu_i^{I}, p_{r,k}^{I})  

$$

  

For the text side, let the assigned prototype index be

  

$$  

a_i^{T} = \arg\max_{k} \operatorname{sim}(\nu_i^{T,\text{exact}}, p_{r,k}^{T})  

$$

  

These assignments define the semantic group membership used by the semantic regularizer.

  

---

  

### 15. Soft prototype targets

  

Instead of using one-hot cluster assignments, the revised method uses probability-based soft targets in order to transfer the relational structure among prototypes.

  

For the image-side prototype set, define

  

$$  

y_k^{I}

=

\operatorname{softmax}\left(S_k^{I}/\tau_y\right)  

\qquad\text{where}\qquad  

S_k^{I} = P_r^{I} p_{r,k}^{I}  

$$

  

For the text-side prototype set, define

  

$$  

y_k^{T}

=

\operatorname{softmax}\left(S_k^{T}/\tau_y\right)  

\qquad\text{where}\qquad  

S_k^{T} = P_r^{T} p_{r,k}^{T}  

$$

  

Then the sample-wise soft targets are

  

$$  

y_i^{I} = y_{a_i^{I}}^{I}  

\qquad\text{and}\qquad  

y_i^{T} = y_{a_i^{T}}^{T}  

$$

  

These targets preserve not only the identity of the assigned cluster but also the neighborhood relations among clusters.

  

---

  

### 16. Prototype Back Translation

  

A core issue in cross-modal prototype supervision is the modality gap. To avoid forcing image representations toward text prototypes in text space or forcing text representations toward image prototypes in image space, the revised method uses Prototype Back Translation.

  

For each text-side prototype $p_{r,k}^{T}$, retrieve all image samples assigned to this text prototype:

  

$$  

\mathcal{I}_k^{T} = {i : a_i^{T} = k}  

$$

  

Then compute the within-image centroid

  

$$  

c_{r,k}^{T \rightarrow I}

=

\frac{1}{|\mathcal{I}_k^{T}|}  

\sum_{i \in \mathcal{I}_k^{T}} \operatorname{sg}(\nu_i^{I})  

$$

  

Similarly, for each image-side prototype $p_{r,k}^{I}$, retrieve all text samples assigned to this image prototype:

  

$$  

\mathcal{I}_k^{I} = {i : a_i^{I} = k}  

$$

  

Then compute the within-text centroid

  

$$  

c_{r,k}^{I \rightarrow T}

=

\frac{1}{|\mathcal{I}_k^{I}|}  

\sum_{i \in \mathcal{I}_k^{I}} \operatorname{sg}(\nu_i^{T,\text{exact}})  

$$

  

Collecting all translated centroids gives

  

$$  

C_r^{T \rightarrow I}

=

\begin{bmatrix}  

c_{r,1}^{T \rightarrow I} \  

c_{r,2}^{T \rightarrow I} \  

\vdots \  

c_{r,N}^{T \rightarrow I}  

\end{bmatrix}  

\in \mathbb{R}^{N \times d_h}  

$$

  

and

  

$$  

C_r^{I \rightarrow T}

=

\begin{bmatrix}  

c_{r,1}^{I \rightarrow T} \  

c_{r,2}^{I \rightarrow T} \  

\vdots \  

c_{r,N}^{I \rightarrow T}  

\end{bmatrix}  

\in \mathbb{R}^{N \times d_h}  

$$

  

These translated centroids are within-modal anchors. Therefore, they regularize representation grouping without requiring strict cross-modal coordinate alignment.

  

---

  

### 17. Semantic structure probabilities

  

Using the translated centroids, we compute prototype classification probabilities for the student representations.

  

For image-side semantic features supervised by text-derived translated centroids:

  

$$  

p_i^{I \leftarrow T}

=

\operatorname{softmax}\left(  

\frac{C_r^{T \rightarrow I} \nu_i^{I}}{\tau_{\text{sem}}}  

\right)  

$$

  

For surrogate-text semantic features supervised by image-derived translated centroids, first define

  

$$  

\nu_i^{T,\text{surr}} = g_t^{\text{proto}}\left(\hat{t}_{i \mid i}^{\text{proto}}\right)  

$$

  

and then compute

  

$$  

p_i^{T \leftarrow I}

=

\operatorname{softmax}\left(  

\frac{C_r^{I \rightarrow T} \nu_i^{T,\text{surr}}}{\tau_{\text{sem}}}  

\right)  

$$

  

So the student objects of the semantic regularizer are:

  

- $\nu_i^{I}$ on the image side

- $\nu_i^{T,\text{surr}}$ on the surrogate-text side

  

while the teacher-side grouping structure is determined by recomputed prototype assignments and their translated within-modal centroids.

  

---

  

### 18. Semantic structure loss

  

The semantic structure loss is defined as a bidirectional soft-target cross-entropy over translated prototype centroids:

  

$$  

\mathcal{L}_{\text{sem}}

=

-\frac{1}{2B}  

\sum_{i=1}^{B}  

\left[  

\sum_{k=1}^{N} y_{i,k}^{T} \log p_{i,k}^{I \leftarrow T}  

+  

\sum_{k=1}^{N} y_{i,k}^{I} \log p_{i,k}^{T \leftarrow I}  

\right]  

$$

  

This loss has the following interpretation:

  

- the image-side semantic feature learns the grouping structure induced by text-side prototypes after back translation into image space

- the surrogate-text semantic feature learns the grouping structure induced by image-side prototypes after back translation into text space

- the supervision is structural rather than retrieval-oriented

- the supervision does not require a second pairwise score matrix

  

This is the main replacement for the old prototype retrieval loss.

  

---

  
  

### 19. Host mined semantic hard negative margin loss (current code path)

  

The semantic structure loss above organizes image side and surrogate text side representations around translated prototype centroids. In addition to this structural grouping term, we use a host mined semantic hard negative margin loss to make the semantic regularizer more sensitive to confusing negative pairs.

  

The key distinction is that deployed host scores are used only for hard negative index mining. The primary hinge values are still computed from semantic or prototype side compatibility terms derived from semantic structure probabilities and soft prototype targets. Therefore, this term remains a semantic regularizer and does not introduce a deployed prototype retrieval score.

  

For each anchor image $i$, the deployed host score matrix is first used to select the hardest negative caption index within the mini batch:

  

$$

j_i^*

=

\arg\max_{j \ne i} s_{ij}^{\text{host}}

$$

  

where $s_{ij}^{\text{host}}$ is the original ITSELF host score defined in Section 4. This mining step is detached from the margin scoring path and only determines which negative caption is used.

  

The semantic margin is then computed from the prototype mediated PBT compatibility terms. Recall that the semantic structure module produces:

  

$$

p_i^{I \leftarrow T}

=

\operatorname{softmax}

\left(

\frac{C_r^{T \rightarrow I}\nu_i^I}{\tau_{\text{sem}}}

\right)

$$

  

and

  

$$

p_i^{T \leftarrow I}

=

\operatorname{softmax}

\left(

\frac{C_r^{I \rightarrow T}\nu_i^{T,\text{surr}}}{\tau_{\text{sem}}}

\right)

$$

  

with soft prototype targets $y_i^T$ and $y_i^I$.

  

For the image side semantic compatibility, define:

  

$$

\phi_{ij}^{I}

=

\sum_{k=1}^{N} y_{j,k}^{T}\log p_{i,k}^{I \leftarrow T}

$$

  

This measures how compatible the image side student distribution of anchor $i$ is with the text side prototype target of caption $j$ after Prototype Back Translation into image space.

  

For the surrogate text side semantic compatibility, define:

  

$$

\phi_{ij}^{T}

=

\sum_{k=1}^{N} y_{j,k}^{I}\log p_{i,k}^{T \leftarrow I}

$$

  

This measures how compatible the surrogate text side student distribution associated with anchor $i$ is with the image side prototype target of sample $j$ after Prototype Back Translation into text space.

  

The positive semantic compatibility terms are:

  

$$

\phi_{ii}^{I}

\qquad\text{and}\qquad

\phi_{ii}^{T}

$$

  

and the host mined negative compatibility terms are:

  

$$

\phi_{i j_i^*}^{I}

\qquad\text{and}\qquad

\phi_{i j_i^*}^{T}

$$

  

Define the semantic hard negative component:

  

$$

\mathcal{L}_{\text{shn-sem}}

=

\frac{1}{2B}

\sum_{i=1}^{B}

\left[

\Delta_s

-

\left(

\phi_{ii}^{I}

-

\phi_{i j_i^*}^{I}

\right)

\right]_+

+

\frac{1}{2B}

\sum_{i=1}^{B}

\left[

\Delta_s

-

\left(

\phi_{ii}^{T}

-

\phi_{i j_i^*}^{T}

\right)

\right]_+

$$

  

where $\Delta_s$ is the target semantic margin and $[x]_+ = \max(0,x)$.

In the current implementation, this semantic component is augmented by a semantic-guided host-global bridge term that still uses the same mined hard pairs.

Let a generic host-visible global score matrix be

$$
g_{ij}^{\text{host-global}}
=
\operatorname{sim}\left(z_i^{v,\text{host-global}}, z_j^{t,\text{host-global}}\right)
$$

where these global embeddings are exposed by the host wrapper (works for both ITSELF and CLIP wrapper paths, without editing host internals).

For the mined hard negative caption index $j_i^*$ and mined hard negative image index $k_i^*$, define:

$$
m_i^{I,\text{host-global}}
=
g_{ii}^{\text{host-global}} - g_{i j_i^*}^{\text{host-global}}
$$

$$
m_i^{T,\text{host-global}}
=
g_{ii}^{\text{host-global}} - g_{k_i^* i}^{\text{host-global}}
$$

Define detached semantic gap gates:

$$
w_i^I
=
\sigma\left(
\frac{\Delta_s - \left(\phi_{ii}^I - \phi_{i j_i^*}^I\right)}{\tau_{\text{hg}}}
\right),\quad
w_i^T
=
\sigma\left(
\frac{\Delta_s - \left(\phi_{ii}^T - \phi_{i j_i^*}^T\right)}{\tau_{\text{hg}}}
\right)
$$

and the bridge term:

$$
\mathcal{L}_{\text{shn-bridge}}
=
\frac{1}{2B}\sum_{i=1}^{B}
\left(
w_i^I \left[\Delta_s - m_i^{I,\text{host-global}}\right]_+

+ 
w_i^T \left[\Delta_s - m_i^{T,\text{host-global}}\right]_+
\right)
$$

The implemented loss is:

$$
\mathcal{L}_{\text{semantic\_hardneg\_margin}}
=
\mathcal{L}_{\text{shn-sem}}
+
\lambda_{\text{hg}}\,\mathcal{L}_{\text{shn-bridge}}
$$

with implementation parameters:

- $\lambda_{\text{hg}}$ = `semantic_hardneg_host_global_weight`
- $\tau_{\text{hg}}$ = `semantic_hardneg_host_global_tau`

  

This term has three important properties.

  

1. The hard negative is selected by the host score $s_{ij}^{\text{host}}$.

  

2. The primary hinge value is semantic/prototype compatibility; the additional host-global bridge uses host-visible global embeddings only, gated by detached semantic gaps.

  

3. The loss is training only and does not create a prototype based inference score.

  

Thus, this term is best understood as a host mined semantic hard negative regularizer with a semantic-guided host-global bridge. It still complements semantic structure training and still keeps deployed retrieval host-only.

Current debug metrics for this path include:

- `sem_hardneg_pos_img_mean`
- `sem_hardneg_neg_img_mean`
- `sem_hardneg_pos_txt_mean`
- `sem_hardneg_neg_txt_mean`
- `sem_hardneg_pos_host_global_mean`
- `sem_hardneg_neg_host_global_mean`
- `sem_hardneg_margin_host_global_mean`
- `sem_hardneg_bridge_weight_mean`
- `sem_hardneg_bridge_loss_mean`

  

---

  

### 20. Final overall objective

  

The final total objective of the integrated method is

  

$$

\mathcal{L}

=

\mathcal{L}_{\text{ITSELF}}

+

\lambda_{\text{diag}}\mathcal{L}_{\text{diag}}

+

\lambda_{\text{sem}}\mathcal{L}_{\text{sem}}

+

\lambda_{\text{shn}}\mathcal{L}_{\text{semantic\_hardneg\_margin}}

$$

where in the current code path $\mathcal{L}_{\text{semantic\_hardneg\_margin}}$ internally includes:

1. a semantic PBT-compatibility hinge component
2. a semantic-gated host-global bridge component

  

with

  

$$

\mathcal{L}_{\text{ITSELF}}

=

\mathcal{L}_{\text{global}}^{\text{host}}

+

\mathcal{L}_{\text{local}}^{\text{host}}

$$

  

The objective contains three major parts.

  

1. The original ITSELF host supervision for retrieval alignment.

  

2. The prototype mediated semantic structure supervision for representation grouping.

  

3. The host mined semantic hard negative supervision for hard negative awareness inside the semantic regularization space.

  

The diagonal loss plays a bridging role because it keeps the surrogate text object semantically faithful to the exact diagonal teacher. The semantic structure loss organizes the geometry around stable semantic anchors. The semantic hard negative margin loss adds a harder pairwise constraint inside the semantic structure module, but it does not replace or modify the deployed host scoring function.

  

---

  

### 21. Training principle

  

The method follows the following training principle.

  

1. The ITSELF host path is preserved as the base retrieval path.

  

2. The prototype subsystem is attached as a semantic organizer rather than a second scorer.

  

3. The prototype bank is recomputed from detached features rather than updated directly by SGD.

  

4. The surrogate text branch remains trainable and is stabilized by diagonal fidelity.

  

5. The host and prototype side projectors can be optimized jointly from the beginning.

  

6. The semantic structure loss acts on representation geometry instead of score fusion.

  

7. The semantic hard negative margin loss uses host mined negatives, keeps semantic PBT compatibility as the main hinge source, and adds a semantic-gated host-global bridge term through generic host-visible global embeddings.

  

This gives a clean conceptual decomposition.

  

1. Host retrieval learning.

  

2. Semantic anchor induction.

  

3. Image conditioned surrogate text construction.

  

4. Semantic grouping regularization.

  

5. Host mined semantic hard negative regularization.

  

---

  

### 22. Joint training direction

  

The main intended training direction is joint optimization from the beginning.

  

Let:

  

1. $\theta_H$ denote host parameters.

  

2. $\theta_S$ denote semantic structure side trainable parameters, including prototype side projectors, routing side transforms if any, and surrogate text construction modules.

  

Then the intended training regime is

  

$$

\theta_H \text{ trainable}, \qquad \theta_S \text{ trainable}

$$

  

while the prototype anchors themselves are not updated by gradient, but only by periodic recomputation.

  

Thus, the network learns how to produce a representation space in which K Means induced anchors become more meaningful over time, while the anchors themselves remain semantic summaries of the current detached feature space.

  

---

  

### 23. Prototype recomputation schedule

  

Let prototype recomputation happen every $\Delta_r$ update units, where the update unit may be an epoch, an episode, or a fixed number of optimization steps.

  

At each recomputation step $r$:

  

1. Extract detached image side semantic features $\nu^{I}$.

  

2. Extract detached exact text semantic features $\nu^{T,\text{exact}}$.

  

3. Run K Means to obtain $P_r^{I}$ and $P_r^{T}$.

  

4. Optionally contextualize them to obtain $\bar{P}_r^{I}$ and $\bar{P}_r^{T}$.

  

5. Keep these anchors fixed until the next recomputation step.

  

This schedule makes semantic structure a living property of the evolving representation space rather than a one time initialization trick.

  

---

  

### 24. Full training pipeline

  

For a mini batch

  

$$

{(x_i, c_i, y_i)}_{i=1}^{B}

$$

  

the training pipeline is as follows.

  

First, the ITSELF host image encoder produces

  

$$

V_i = f_v^{\text{host}}(x_i)

=

{v_i^{\text{global}}, v_i^{\text{local}}}

$$

  

Second, the ITSELF host text encoder produces

  

$$

T_j = f_t^{\text{host}}(c_j)

=

{t_j^{\text{global}}, t_j^{\text{local}}}

$$

  

with token states

  

$$

H_j = [h_{j,1}, \dots, h_{j,L_j}]

$$

  

Third, the ITSELF host path computes its own global and local features and its host loss

  

$$

\mathcal{L}_{\text{ITSELF}}

=

\mathcal{L}_{\text{global}}^{\text{host}}

+

\mathcal{L}_{\text{local}}^{\text{host}}

$$

  

Fourth, the current recomputed prototype anchors $P_r^{I}$ and $P_r^{T}$ are loaded, and optionally contextualized to obtain $\bar{P}_r^{I}$ and $\bar{P}_r^{T}$.

  

Fifth, each image computes routing weights

  

$$

\alpha_{i,n}

=

\frac{

\exp\left(\operatorname{sim}(\nu_i^{I}, \bar{p}_{r,n}^{I})/\tau_p\right)

}{

\sum_{m=1}^{N}\exp\left(\operatorname{sim}(\nu_i^{I}, \bar{p}_{r,m}^{I})/\tau_p\right)

}

$$

  

Sixth, each image forms its semantic summary

  

$$

q_i = \sum_{n=1}^{N}\alpha_{i,n}\bar{p}_{r,n}^{I}

$$

  

Seventh, for each caption and each text side prototype, the caption basis bank is built:

  

$$

b_{j,n}

=

\sum_{\ell=1}^{L_j}

\gamma_{j,n,\ell}h_{j,\ell}

$$

  

with

  

$$

\gamma_{j,n,\ell}

=

\frac{

\exp\left((\bar{p}_{r,n}^{T})^\top h_{j,\ell}/\tau_b\right)

}{

\sum_{r=1}^{L_j}\exp\left((\bar{p}_{r,n}^{T})^\top h_{j,r}/\tau_b\right)

}

$$

  

Eighth, on the diagonal or matched construction path, the surrogate prototype text object is constructed

  

$$

\hat{t}_{i \mid i}^{\text{proto}}

=

\sum_{n=1}^{N}

\alpha_{i,n}b_{i,n}

$$

  

and projected

  

$$

\hat{z}_{i \mid i}^{t,\text{proto}}

=

g_t^{\text{proto}}\left(\hat{t}_{i \mid i}^{\text{proto}}\right)

$$

  

Ninth, the exact diagonal teacher is built

  

$$

t_{i \mid i}^{\text{exact}}

=

\sum_{\ell=1}^{L_i}

\beta_{i,\ell}^{\text{exact}}h_{i,\ell}

$$

  

and projected to $z_{i \mid i}^{t,\text{exact}}$.

  

Tenth, the diagonal fidelity loss $\mathcal{L}_{\text{diag}}$ is computed.

  

Eleventh, image side and exact text side semantic features are assigned to the current base prototypes, the soft targets are formed, PBT translated centroids are computed, and the semantic structure loss $\mathcal{L}_{\text{sem}}$ is evaluated.

  

Twelfth, the host score matrix $s_{ij}^{\text{host}}$ is used only to select hardest negative indices. The selected negatives are scored by semantic PBT compatibility terms, and a semantic-gated host-global bridge margin term is also applied on the same mined pairs, producing $\mathcal{L}_{\text{semantic\_hardneg\_margin}}$.

  

Thirteenth, the full loss

  

$$

\mathcal{L}

=

\mathcal{L}_{\text{ITSELF}}

+

\lambda_{\text{diag}}\mathcal{L}_{\text{diag}}

+

\lambda_{\text{sem}}\mathcal{L}_{\text{sem}}

+

\lambda_{\text{shn}}\mathcal{L}_{\text{semantic\_hardneg\_margin}}

$$

  

is optimized.

  

---

  

### 25. Full inference pipeline

  

The deployed inference semantics are simpler than in the earlier retrieval fusion formulation.

  

For each image $x_i$, the model computes:

  

$$

V_i = f_v^{\text{host}}(x_i)

=

{v_i^{\text{global}}, v_i^{\text{local}}}

$$

  

The host then computes its original host score components:

  

$$

s_{ij}^{\text{host}}

=

\lambda_s s_{ij}^{\text{global,host}}

+

(1-\lambda_s)s_{ij}^{\text{local,host}}

$$

  

For a candidate caption $c_j$, the host text encoder produces token states

  

$$

H_j = [h_{j,1}, \dots, h_{j,L_j}]

$$

  

The prototype subsystem may still be executed at inference time if one wants the surrogate text object for analysis, diagnostics, or auxiliary downstream use. In that case, routing weights $\alpha_i$, the basis bank $B_j$, and the surrogate object $\hat{t}_{j \mid i}^{\text{proto}}$ can still be constructed exactly as in training.

  

However, the deployed retrieval score remains

  

$$

s_{ij}^{\text{host}}

$$

  

rather than a fused host plus prototype score.

  

The semantic hard negative margin loss is also training only. At inference time, no semantic hard negative margin score is computed for retrieval, and the deployed ranking still uses only $s_{ij}^{\text{host}}$.

  

Thus, under the integrated formulation:

  

1. The ITSELF host path remains the sole retrieval scorer.

  

2. The prototype subsystem shapes the representation space during training.

  

3. The surrogate text object remains available as a semantic construction object.

  

4. The prototype subsystem is primarily a training time semantic regularizer rather than a second deployed scoring path.

---

### 26. SPB-K: Sufficient Prototype Basis K Selection

#### 26.1 Motivation: prototype number as semantic basis width

The number of prototypes is not merely the number of K-Means clusters. In the present architecture, the same quantity also controls the width of the prototype-conditioned text basis bank. For a caption $c_j$, the prototype text branch constructs

$$
B_j(K)=
\begin{bmatrix}
b_{j,1} \\
b_{j,2} \\
\vdots \\
b_{j,K}
\end{bmatrix}
\in \mathbb{R}^{K\times D},
$$

where $b_{j,k}$ is the text basis vector induced by the $k$-th text-side semantic anchor. The image branch produces a routing distribution

$$
\alpha_i^K\in\Delta^{K-1},
$$

and the image-conditioned surrogate text object is written as

$$
\hat t_{j\mid i,K}^{\mathrm{proto}}
=
\sum_{k=1}^{K}\alpha_{i,k}^{K} b_{j,k}.
$$

Therefore, changing $K$ changes more than the semantic partition. It changes the routing simplex, the image-conditioned summary, the number of caption basis rows, the surrogate text construction path, the diagonal fidelity geometry, and the support structure used by Prototype Back Translation. A larger $K$ does not simply mean a stronger model. It may introduce redundant basis rows, low-support clusters, unstable routing, and noisy translated centroids.

This means that $K$ should be selected according to the role prototypes play in Version 3. The prototype branch is a training-time semantic structure module, not an auxiliary retrieval scorer. The correct question is not:

> How many semantic clusters exist in the dataset?

The better question is:

> How many prototype-conditioned text bases are sufficient for stable image-conditioned surrogate text construction?

SPB-K, or **Sufficient Prototype Basis K selection**, answers this question directly.

---

#### 26.2 Definition of $K$ under SPB-K

Under SPB-K, $K$ is the **semantic basis width** of the prototype-mediated surrogate construction module. It has two linked meanings.

First, $K$ is the number of detached semantic anchors induced by K-Means:

$$
P_r^I(K)=\operatorname{KMeans}_{K}\big(\operatorname{sg}(\nu^I)\big),
$$

$$
P_r^T(K)=\operatorname{KMeans}_{K}\big(\operatorname{sg}(\nu^{T,\mathrm{exact}})\big),
$$

where $r$ denotes the prototype recomputation step and $\operatorname{sg}(\cdot)$ denotes stop-gradient. The image-side anchors define routing over semantic structure, and the text-side anchors define prototype-conditioned views of each caption.

Second, $K$ is the number of rows in the caption-specific basis bank:

$$
B_j(K)\in\mathbb{R}^{K\times D}.
$$

This bank is mixed by image routing to produce the surrogate text object. Thus, $K$ controls both the resolution of detached semantic anchors and the width of the basis through which image-conditioned information is written into text.

A good $K$ should be large enough to provide expressive semantic bases, but small enough to avoid over-fragmented anchors and redundant basis rows. The desired value is the smallest basis width that is sufficient for the surrogate construction role.

---

#### 26.3 Principle: smallest sufficient semantic basis

SPB-K follows a smallest-sufficient-width principle:

> Select the smallest feasible $K$ whose no-gradient diagonal calibration fidelity is statistically indistinguishable from the best candidate.

The principle intentionally favors sufficiency over maximal granularity. If $K=64$ is only marginally better than $K=32$, or not better at all, then $K=32$ should be preferred because it gives a simpler and more stable basis bank. This is not a preference for a smaller model. It follows from the architecture: additional prototypes are useful only if the additional basis rows improve surrogate construction while remaining well used and well supported.

The selector is intended to run once per dataset after a short warm-up. Once selected,

$$
K^\star
$$

is fixed for the remaining training process. Periodic recomputation still updates the detached anchor locations, but it does not repeatedly change the number of anchors:

$$
P_r^I=\operatorname{KMeans}_{K^\star}\big(\operatorname{sg}(\nu^I)\big),
$$

$$
P_r^T=\operatorname{KMeans}_{K^\star}\big(\operatorname{sg}(\nu^{T,\mathrm{exact}})\big).
$$

This keeps the method simple and paper-friendly: SPB-K adapts the dataset-level semantic basis width, not the prototype count at every recomputation.

---

#### 26.4 Candidate set

SPB-K uses a small candidate set. For the current computational regime, the natural choice is

$$
\mathcal{K}=\{16,32,64\}.
$$

A smaller value such as $K=8$ can be included as a diagnostic, but the main method should remain simple. The candidate set should cover the plausible range where the basis bank is neither obviously under-complete nor obviously too expensive.

For each $K\in\mathcal{K}$, the model temporarily instantiates candidate detached prototypes, constructs the corresponding routing distributions and text basis banks, and evaluates the diagonal surrogate construction without gradient updates. This is not a validation-set retrieval search. It does not use Rank-1, mAP, prototype retrieval similarity, or host failure cases to choose $K$.

---

#### 26.5 Diagonal calibration fidelity

The main calibration metric is the existing diagonal fidelity objective. This is the right signal because the surrogate text object remains central to the method, and diagonal fidelity measures whether the surrogate object is semantically tied to the exact diagonal teacher.

For candidate $K$, the prototype branch constructs

$$
\hat z_{i\mid i,K}^{t,\mathrm{proto}}
=
g_t^{\mathrm{proto}}\big(\hat t_{i\mid i,K}^{\mathrm{proto}}\big),
$$

and the exact diagonal teacher

$$
z_{i\mid i,K}^{t,\mathrm{exact}}
=
g_t^{\mathrm{proto}}\big(t_{i\mid i,K}^{\mathrm{exact}}\big).
$$

The diagonal relation matrix is

$$
M_{ik}^{K}
=
\cos\left(
\hat z_{i\mid i,K}^{t,\mathrm{proto}},
\operatorname{sg}\left(z_{k\mid k,K}^{t,\mathrm{exact}}\right)
\right).
$$

Using the same symmetric diagonal relation objective as in training, define the no-gradient calibration loss:

$$
\mathcal{L}_{\mathrm{diag}}^{\mathrm{cal}}(K)
=
\mathcal{L}_{\mathrm{diag-rel}}^{\mathrm{sym}}(K).
$$

Across calibration mini-batches,

$$
\mu_K=\operatorname{mean}\left(\mathcal{L}_{\mathrm{diag}}^{\mathrm{cal}}(K)\right),
$$

and

$$
SE_K=
\frac{\operatorname{std}\left(\mathcal{L}_{\mathrm{diag}}^{\mathrm{cal}}(K)\right)}
{\sqrt{B_{\mathrm{cal}}}},
$$

where $B_{\mathrm{cal}}$ is the number of calibration batches. A smaller $\mu_K$ means the $K$-wide basis bank constructs a surrogate text object that is more faithful to the exact diagonal teacher.

This criterion is aligned with the architecture because it evaluates the quality of the training-time construction path. It does not evaluate a prototype retrieval score.

---

#### 26.6 Routing usage

Diagonal fidelity alone is not enough. A large $K$ may give similar calibration loss while using only a small subset of its basis rows. Such a candidate should not be preferred, because the nominal basis width is not actually being used.

For candidate $K$, let $\alpha_i^K$ be the image routing distribution. Define the mean routing mass

$$
\bar\alpha^K
=
\frac{1}{N_{\mathrm{cal}}}\sum_i\alpha_i^K.
$$

The active basis count is

$$
A(K)=\exp\left(H(\bar\alpha^K)\right),
$$

where

$$
H(\bar\alpha^K)
=
-
\sum_{k=1}^{K}\bar\alpha_k^K\log \bar\alpha_k^K.
$$

The normalized routing usage is

$$
U(K)=\frac{A(K)}{K}.
$$

A candidate is feasible only if

$$
U(K)\geq \rho,
$$

with a typical default

$$
\rho=0.5.
$$

This rejects a large basis bank whose effective routing behavior is much smaller than its nominal width.

---

#### 26.7 Cluster support

SPB-K also requires the detached K-Means anchors to have enough support. This matters because prototypes define assignments, soft prototype targets, and PBT translated centroids. Tiny clusters can produce noisy centroids and unstable structural supervision.

For image-side assignments, let $n_k^I$ be the number of samples assigned to image prototype $k$. Define

$$
\operatorname{p10}^I(K)
=
\operatorname{quantile}_{0.10}\left(\{n_k^I\}_{k=1}^{K}\right).
$$

If text-side assignments are available, define

$$
\operatorname{p10}^T(K)
=
\operatorname{quantile}_{0.10}\left(\{n_k^T\}_{k=1}^{K}\right).
$$

A candidate is support-feasible only if

$$
\operatorname{p10}^I(K)\geq s_{\min},
$$

and, when text-side statistics exist,

$$
\operatorname{p10}^T(K)\geq s_{\min}.
$$

A practical value is

$$
s_{\min}=4.
$$

This condition is intentionally simple. It only removes candidates whose anchors are too poorly supported to serve as stable routing, basis, and PBT structures.

---

#### 26.8 Feasible set and selection rule

The feasible candidate set is

$$
\mathcal{F}
=
\left\{
K\in\mathcal{K}:
U(K)\geq\rho,
\operatorname{p10}^I(K)\geq s_{\min},
\operatorname{p10}^T(K)\geq s_{\min}\ \text{if available}
\right\}.
$$

If no candidate is feasible, SPB-K keeps the current configured value:

$$
K^\star=K_{\mathrm{current}}.
$$

This fallback avoids changing training when the calibration evidence is unreliable.

If feasible candidates exist, define

$$
K_{\mathrm{best}}
=
\arg\min_{K\in\mathcal{F}}\mu_K.
$$

With the one-standard-error rule, define

$$
\theta=
\mu_{K_{\mathrm{best}}}+SE_{K_{\mathrm{best}}}.
$$

Then choose

$$
K^\star
=
\min\left\{
K\in\mathcal{F}:\mu_K\leq\theta
\right\}.
$$

In words:

> Choose the smallest feasible basis width whose diagonal calibration fidelity is statistically tied with the best feasible candidate.

If the one-standard-error rule is disabled, choose

$$
K^\star=K_{\mathrm{best}}.
$$

The one-standard-error rule is preferred because it matches the purpose of SPB-K: find a sufficient semantic basis width, not the largest stable cluster count.

---

#### 26.9 Expected behavior and interpretation

SPB-K explains why a moderate value such as $K=32$ can outperform $K=64$.

When $K$ is too small, the basis bank may be under-complete. Many distinct appearance or caption patterns must share too few basis rows. Routing usage may be high, but diagonal calibration fidelity can be weak because the surrogate text object lacks sufficient semantic degrees of freedom.

When $K$ is too large, the basis bank may become over-fragmented. Additional rows may attend to similar caption tokens, receive low routing mass, or depend on low-support clusters. Calibration fidelity may saturate or degrade, while routing usage and cluster support decline.

A typical pattern is:

| Candidate $K$ | Expected behavior | Interpretation |
|---:|---|---|
| 16 | high usage, weaker diagonal fidelity | basis may be insufficient |
| 32 | strong fidelity, healthy usage, adequate support | sufficient semantic basis width |
| 64 | similar or worse fidelity, lower usage, lower support | redundant or fragmented basis bank |

If $K=64$ is only marginally better than $K=32$ in calibration fidelity, SPB-K chooses $K=32$. The goal is not to maximize the number of prototypes. The goal is to identify the smallest sufficient semantic basis.

---

#### 26.10 Relationship to periodic prototype recomputation

SPB-K and periodic prototype recomputation solve different problems.

Periodic recomputation updates the locations of the detached anchors as the representation space evolves:

$$
P_r^I, P_r^T \quad \text{change over training}.
$$

SPB-K chooses the dataset-level basis width:

$$
K^\star \quad \text{is selected once and then held fixed}.
$$

After selection, subsequent recomputation uses the selected width but still recomputes the anchors from detached features:

$$
P_r^I=\operatorname{KMeans}_{K^\star}\big(\operatorname{sg}(\nu^I)\big),
$$

$$
P_r^T=\operatorname{KMeans}_{K^\star}\big(\operatorname{sg}(\nu^{T,\mathrm{exact}})\big).
$$

Thus, prototypes remain detached K-Means anchors, not learnable parameters updated directly by SGD. SPB-K changes only how the dataset-level width is chosen.

---

#### 26.11 Training and inference separation

SPB-K is strictly training-time. It does not create a prototype retrieval score and does not alter the deployed retrieval path.

At inference, the retrieval score remains the host score:

$$
s_{ij}^{\mathrm{host}}
=
\lambda_s s_{ij}^{\mathrm{global,host}}
+
(1-\lambda_s)s_{ij}^{\mathrm{local,host}}.
$$

There is no inference-time query-prototype similarity, no prototype-enhanced ranking score, and no host-prototype score fusion:

$$
s_{ij}^{\mathrm{final}}
\neq
s_{ij}^{\mathrm{host}}+\lambda s_{ij}^{\mathrm{proto}}.
$$

The selected $K^\star$ only determines the width of the training-time prototype-mediated surrogate construction and semantic regularization pathway. The deployed retrieval scorer remains host-only.

---

#### 26.12 Distinction from density-based adaptive cluster estimation

Density-based adaptive prototype methods estimate the number of prototypes by identifying density modes in feature space. Such methods answer:

> How many prominent feature-density modes exist?

SPB-K answers a different question:

> How many prototype-conditioned text bases are sufficient for stable surrogate construction?

This distinction matters because a dataset may contain many density modes while the basis bank needs only a smaller number of stable bases. Conversely, compact text distributions may still require enough basis rows to express image-conditioned surrogate variation.

Density-based selection can remain a diagnostic or appendix-level comparison. The main method should be SPB-K because it is tied to the actual objects used by the model: routing weights, basis rows, diagonal surrogate embeddings, exact diagonal teachers, and supported detached anchors.

| Method type | Main question | Role of prototypes | Inference behavior |
|---|---|---|---|
| Density-based adaptive clustering | How many density modes exist? | feature-space cluster centers | may support retrieval refinement in other methods |
| SPB-K | How many text bases are sufficient? | training-time semantic basis and structure anchors | no prototype scoring; host-only retrieval |

---

#### 26.13 Paper-ready formulation

The method can be written as follows:

> The number of prototypes controls not only the number of K-Means anchors, but also the width of the prototype-conditioned text basis bank used to construct the image-conditioned surrogate text object. Therefore, a larger $K$ is not necessarily better: it may introduce redundant basis rows, low-support clusters, unstable routing, and noisy PBT centroids. We propose Sufficient Prototype Basis K selection, or SPB-K, to choose the dataset-level prototype width. After a short warm-up, each candidate $K$ is evaluated without gradient by instantiating detached K-Means anchors and measuring diagonal calibration fidelity between the surrogate text object and the exact diagonal teacher. A candidate is feasible only if its routing usage is non-degenerate and its image/text cluster supports are sufficient. Among feasible candidates, we choose the smallest $K$ whose diagonal calibration loss is within one standard error of the best candidate. The selected $K$ is then fixed for all subsequent prototype recomputations. This adapts the semantic basis width of the training-time geometry regularizer without introducing prototype-based inference scoring.

The one-sentence framing is:

> SPB-K adapts the sufficient semantic basis width of the prototype-mediated surrogate construction module, not the capacity of a retrieval expert.

---

#### 26.14 Recommended experimental reporting

The main empirical claim should be that $K$ has a sufficiency point rather than a monotonic scaling behavior. The main comparison should use the fixed grid

$$
K\in\{16,32,64\}
$$

and compare it with SPB-K.

For each dataset, report the final retrieval performance using the host-only inference score. The retrieval metrics should evaluate the selected model, but they should not be used to select $K$. The selection itself should be justified by:

1. diagonal calibration loss $\mu_K$,
2. standard error $SE_K$,
3. routing usage $U(K)$,
4. active basis count $A(K)$,
5. image-side p10 cluster support,
6. text-side p10 cluster support when available,
7. selected value $K^\star$.

A concise reporting table can show why the selected value is sufficient:

| $K$ | Diagonal calibration loss | Routing usage | Image p10 support | Text p10 support | Feasible | Selected |
|---:|---:|---:|---:|---:|:---:|:---:|
| 16 | higher | high | high | high | yes | no |
| 32 | near best | healthy | adequate | adequate | yes | yes |
| 64 | best or tied | lower | lower | lower | yes/no | no |

This table supports the methodological claim that SPB-K chooses a sufficient basis width rather than a larger prototype count.

---

#### 26.15 Failure cases and safeguards

SPB-K is simple, but several cases should be monitored.

If selection is performed too early, the semantic features may not yet be meaningful. A short warm-up should be used before calibration.

If all candidates have similar diagonal fidelity, the one-standard-error rule will choose the smallest feasible value. This is acceptable when routing usage and support are healthy, but it should be reported transparently.

If a larger $K$ gives better diagonal fidelity but much worse routing usage or support, it should be rejected because the basis bank is structurally degenerate.

If no candidate satisfies the feasibility constraints, the method should retain the current configured value of $K$. This prevents unreliable calibration from disrupting training.

If the routing temperature or basis construction mechanism changes substantially, SPB-K should be re-evaluated because the meaning of routing usage may change.

---

#### 26.16 Final interpretation

SPB-K is the main adaptive $K$ mechanism for the current Version 3 geometry regularizer.

It is simple by design. It does not try to discover all semantic modes in the dataset. It does not use prototypes as inference-time retrieval anchors. It does not introduce a second decision path. It does not cluster host failures. It does not repair host boundaries through residual supervision.

Instead, it selects

$$
K^\star
=
\text{the smallest feasible semantic basis width statistically tied with the best diagonal calibration fidelity}.
$$

This is aligned with the architecture because the prototype subsystem is a training-time semantic structure module. Its role is to provide detached anchors, routing distributions, text basis banks, surrogate text construction, diagonal fidelity, and PBT semantic regularization. The selected $K^\star$ determines the sufficient width of that semantic basis system, while the deployed retrieval scorer remains the host alone.
